"""Shared Pytest Fixtures for CFCL SEC Optimization Engine Test Suite.

Provides standardized DCS telemetry payloads, dataframes, model instances,
and API test client fixtures conforming to CFCL plant operational standards.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Generator

import numpy as np
import pandas as pd
import pytest
from starlette.testclient import TestClient

from src.api.main import app
from src.optimization.constrained_optimizer import ConstrainedSECOptimizer
from src.optimization.savings_calculator import SavingsCalculator
from src.utils.config_loader import get_paths, get_project_root


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return root directory of the project."""
    return get_project_root()


@pytest.fixture(scope="session")
def project_paths() -> Dict[str, Path]:
    """Return project directory paths dictionary."""
    return get_paths()


@pytest.fixture(scope="session")
def valid_telemetry_payload() -> Dict[str, Any]:
    """Standard valid DCS telemetry payload for Gadepan-III."""
    return {
        "plant_id": "gadepan_3",
        "timestamp": "2026-06-30T14:00:00",
        "steam_carbon_ratio": 3.05,
        "reformer_outlet_temp": 812.5,
        "loop_pressure": 158.0,
        "ambient_temp": 42.0,
        "ambient_humidity": 38.0,
        "cw_inlet_temp": 33.5,
        "ng_calorific_value": 8620.0,
        "co2_slip_ppm": 450.0,
        "catalyst_age_days": 180.0,
        "shift": "afternoon",
    }


@pytest.fixture(scope="session")
def unsafe_telemetry_payload() -> Dict[str, Any]:
    """Unphysical telemetry payload violating catalyst coking limits (S/C < 2.50)."""
    return {
        "plant_id": "gadepan_3",
        "timestamp": "2026-06-30T14:00:00",
        "steam_carbon_ratio": 1.20,  # Physically dangerous coking condition
        "reformer_outlet_temp": 812.5,
        "loop_pressure": 158.0,
        "ambient_temp": 42.0,
        "ambient_humidity": 38.0,
        "cw_inlet_temp": 33.5,
        "ng_calorific_value": 8620.0,
        "co2_slip_ppm": 450.0,
        "catalyst_age_days": 180.0,
        "shift": "afternoon",
    }


@pytest.fixture(scope="session")
def sample_telemetry_series(valid_telemetry_payload: Dict[str, Any]) -> pd.Series:
    """Preprocessed Series representing a single hour of DCS telemetry."""
    data = valid_telemetry_payload.copy()
    data["hour"] = 14
    data["month"] = 6
    data["hour_sin"] = np.sin(2 * np.pi * 14 / 24.0)
    data["hour_cos"] = np.cos(2 * np.pi * 14 / 24.0)
    data["month_sin"] = np.sin(2 * np.pi * 6 / 12.0)
    data["month_cos"] = np.cos(2 * np.pi * 6 / 12.0)
    data["is_weekend"] = 0
    data["plant_gadepan_1"] = 0.0
    data["plant_gadepan_2"] = 0.0
    data["plant_gadepan_3"] = 1.0
    data["shift_morning"] = 0.0
    data["shift_afternoon"] = 1.0
    data["shift_night"] = 0.0
    return pd.Series(data)


@pytest.fixture(scope="session")
def sample_raw_dataframe() -> pd.DataFrame:
    """Sample raw 24-hour multi-plant telemetry dataframe."""
    np.random.seed(42)
    records = []
    plants = ["gadepan_1", "gadepan_2", "gadepan_3"]
    shifts = ["morning", "afternoon", "night"]

    for i in range(24):
        p = plants[i % 3]
        s = shifts[(i // 8) % 3]
        records.append({
            "timestamp": pd.Timestamp("2026-06-01") + pd.Timedelta(hours=i),
            "plant_id": p,
            "steam_carbon_ratio": np.random.uniform(2.95, 3.15),
            "reformer_outlet_temp": np.random.uniform(800.0, 825.0),
            "loop_pressure": np.random.uniform(150.0, 165.0),
            "ambient_temp": np.random.uniform(30.0, 44.0),
            "ambient_humidity": np.random.uniform(30.0, 60.0),
            "cw_inlet_temp": np.random.uniform(30.0, 35.0),
            "ng_calorific_value": np.random.uniform(8500.0, 8800.0),
            "co2_slip_ppm": np.random.uniform(200.0, 600.0),
            "catalyst_age_days": 100.0 + i,
            "shift": s,
            "sec_gcal_per_mt": 4.95 if p == "gadepan_3" else 5.14,
        })
    return pd.DataFrame(records)


@pytest.fixture(scope="session")
def optimizer_instance() -> ConstrainedSECOptimizer:
    """Preloaded SLSQP constrained optimizer instance."""
    return ConstrainedSECOptimizer()


@pytest.fixture(scope="session")
def savings_calculator_instance() -> SavingsCalculator:
    """Calibrated economics savings calculator instance."""
    return SavingsCalculator()


@pytest.fixture(scope="session")
def api_client() -> Generator[TestClient, None, None]:
    """FastAPI TestClient with lifespan context initialized."""
    with TestClient(app) as client:
        yield client
