"""Centralized configuration loader for the CFCL SEC Optimization pipeline.

Loads and validates YAML configuration files (plant, model, dashboard)
with thread-safe caching to avoid redundant disk reads across modules.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Optional

import yaml


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Resolve project root relative to this file's location:
#   src/utils/config_loader.py  →  ../../  →  project root
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
_CONFIG_DIR: Path = _PROJECT_ROOT / "config"

_REQUIRED_CONFIG_FILES: dict[str, str] = {
    "plant": "plant_config.yaml",
    "model": "model_config.yaml",
    "dashboard": "dashboard_config.yaml",
}

# Thread-safe config cache
_cache: dict[str, dict] = {}
_cache_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ConfigError(Exception):
    """Raised when a configuration file is missing, malformed, or invalid."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_path(config_name: str, config_dir: Optional[Path] = None) -> Path:
    """Resolve the full path to a named config file.

    Parameters
    ----------
    config_name : str
        One of ``"plant"``, ``"model"``, or ``"dashboard"``.
    config_dir : Path, optional
        Override the default config directory.

    Returns
    -------
    Path
        Absolute path to the YAML file.

    Raises
    ------
    ConfigError
        If *config_name* is not recognised or the file does not exist.
    """
    if config_name not in _REQUIRED_CONFIG_FILES:
        raise ConfigError(
            f"Unknown config '{config_name}'. "
            f"Valid names: {list(_REQUIRED_CONFIG_FILES.keys())}"
        )

    base_dir = config_dir or _CONFIG_DIR
    filepath = base_dir / _REQUIRED_CONFIG_FILES[config_name]

    if not filepath.is_file():
        raise ConfigError(f"Config file not found: {filepath}")
    return filepath


def _load_yaml(filepath: Path) -> dict:
    """Read and parse a single YAML file.

    Uses ``yaml.safe_load`` to prevent arbitrary code execution.
    """
    try:
        with open(filepath, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML parse error in {filepath}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(
            f"Expected a YAML mapping (dict) at root of {filepath}, "
            f"got {type(data).__name__}."
        )
    return data


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_plant_config(cfg: dict) -> None:
    """Ensure the plant config has the mandatory structure."""
    if "plants" not in cfg:
        raise ConfigError("plant_config.yaml missing top-level 'plants' key.")
    plants = cfg["plants"]
    if not isinstance(plants, dict) or len(plants) == 0:
        raise ConfigError("plant_config.yaml 'plants' must be a non-empty mapping.")

    for plant_id, plant_data in plants.items():
        for required_key in ("name", "capacity_mtpd", "technology", "sec_range"):
            if required_key not in plant_data:
                raise ConfigError(
                    f"Plant '{plant_id}' missing required key '{required_key}'."
                )
        # SEC range must have min, max, benchmark
        sec = plant_data["sec_range"]
        for k in ("min", "max", "benchmark"):
            if k not in sec:
                raise ConfigError(
                    f"Plant '{plant_id}' sec_range missing '{k}'."
                )


def _validate_model_config(cfg: dict) -> None:
    """Ensure the model config has required sections."""
    if "models" not in cfg:
        raise ConfigError("model_config.yaml missing top-level 'models' key.")
    if "training" not in cfg:
        raise ConfigError("model_config.yaml missing top-level 'training' key.")
    for model_name, model_cfg in cfg["models"].items():
        if "params" not in model_cfg:
            raise ConfigError(
                f"Model '{model_name}' missing 'params' section."
            )


def _validate_dashboard_config(cfg: dict) -> None:
    """Ensure the dashboard config has required sections."""
    if "app" not in cfg:
        raise ConfigError("dashboard_config.yaml missing 'app' key.")
    if "pages" not in cfg:
        raise ConfigError("dashboard_config.yaml missing 'pages' key.")


_VALIDATORS: dict[str, Any] = {
    "plant": _validate_plant_config,
    "model": _validate_model_config,
    "dashboard": _validate_dashboard_config,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(
    config_name: str,
    *,
    config_dir: Optional[Path] = None,
    use_cache: bool = True,
    validate: bool = True,
) -> dict:
    """Load a named configuration file.

    Parameters
    ----------
    config_name : str
        One of ``"plant"``, ``"model"``, or ``"dashboard"``.
    config_dir : Path, optional
        Override the default ``config/`` directory.
    use_cache : bool
        If ``True`` (default), return a cached copy on subsequent calls.
    validate : bool
        If ``True`` (default), run structural validation after loading.

    Returns
    -------
    dict
        Parsed YAML configuration.

    Raises
    ------
    ConfigError
        If the file is missing, malformed, or fails validation.
    """
    cache_key = f"{config_name}:{config_dir or _CONFIG_DIR}"

    if use_cache:
        with _cache_lock:
            if cache_key in _cache:
                return _cache[cache_key]

    filepath = _resolve_path(config_name, config_dir)
    cfg = _load_yaml(filepath)

    if validate:
        validator = _VALIDATORS.get(config_name)
        if validator:
            validator(cfg)

    if use_cache:
        with _cache_lock:
            _cache[cache_key] = cfg

    return cfg


def get_plant_config(plant_id: str, **kwargs) -> dict:
    """Convenience: return config for a single plant by ID.

    Parameters
    ----------
    plant_id : str
        One of ``"gadepan_1"``, ``"gadepan_2"``, ``"gadepan_3"``.

    Returns
    -------
    dict
        Plant-level config subtree.

    Raises
    ------
    ConfigError
        If the plant ID is not found.
    """
    cfg = load_config("plant", **kwargs)
    plants = cfg["plants"]
    if plant_id not in plants:
        raise ConfigError(
            f"Plant '{plant_id}' not found. "
            f"Available: {list(plants.keys())}"
        )
    return plants[plant_id]


def get_process_variable_bounds(variable: str, **kwargs) -> dict:
    """Return safety bounds for a single process variable.

    Parameters
    ----------
    variable : str
        Variable name, e.g. ``"steam_carbon_ratio"``.

    Returns
    -------
    dict
        Contains ``safety_min``, ``safety_max``, and other metadata.
    """
    cfg = load_config("plant", **kwargs)
    pv = cfg.get("process_variables", {})
    if variable not in pv:
        raise ConfigError(
            f"Process variable '{variable}' not found in plant config."
        )
    return pv[variable]


def get_all_process_bounds(**kwargs) -> dict[str, tuple[float, float]]:
    """Return ``{variable_name: (safety_min, safety_max)}`` for all variables."""
    cfg = load_config("plant", **kwargs)
    pv = cfg.get("process_variables", {})
    return {
        name: (spec["safety_min"], spec["safety_max"])
        for name, spec in pv.items()
    }


def get_model_params(model_name: str, **kwargs) -> dict:
    """Return hyperparameters for a named model.

    Parameters
    ----------
    model_name : str
        One of ``"xgboost"``, ``"lstm"``, ``"random_forest"``, ``"linear_baseline"``.
    """
    cfg = load_config("model", **kwargs)
    models = cfg["models"]
    if model_name not in models:
        raise ConfigError(
            f"Model '{model_name}' not found. "
            f"Available: {list(models.keys())}"
        )
    return models[model_name]["params"]


def get_paths(**kwargs) -> dict[str, Path]:
    """Return resolved output paths from model_config.yaml.

    Returns
    -------
    dict
        Mapping of path names to absolute ``Path`` objects.
    """
    cfg = load_config("model", **kwargs)
    raw_paths = cfg.get("paths", {})
    return {
        name: (_PROJECT_ROOT / rel_path).resolve()
        for name, rel_path in raw_paths.items()
    }


def clear_cache() -> None:
    """Clear all cached configurations (useful for testing)."""
    with _cache_lock:
        _cache.clear()


def get_project_root() -> Path:
    """Return the resolved project root directory."""
    return _PROJECT_ROOT
