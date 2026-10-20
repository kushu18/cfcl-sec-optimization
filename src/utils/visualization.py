"""CFCL-branded visualization utilities for the SEC Optimization pipeline.

Provides a consistent, publication-quality plot theme across all figures:
- Corporate colour palette (CFCL blue, plant-specific colours)
- Standardised axis labels, grids, fonts, and sizing
- Reusable helper functions for common plot types

Usage
-----
::

    from src.utils.visualization import apply_cfcl_theme, PLANT_COLORS

    apply_cfcl_theme()            # Call once at script start
    fig, ax = plt.subplots()
    ax.plot(x, y, color=PLANT_COLORS["gadepan_1"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Colour Palette
# ---------------------------------------------------------------------------

# CFCL corporate colours (derived from public branding)
CFCL_BLUE: str = "#1B4F72"
CFCL_LIGHT_BLUE: str = "#2E86C1"
CFCL_DARK: str = "#2C3E50"

# Plant-specific colours (consistent across all visualizations)
PLANT_COLORS: dict[str, str] = {
    "gadepan_1": "#E74C3C",   # Red   — oldest plant
    "gadepan_2": "#3498DB",   # Blue  — mid-generation
    "gadepan_3": "#2ECC71",   # Green — newest, most efficient
}

PLANT_LABELS: dict[str, str] = {
    "gadepan_1": "Gadepan-I (MEA)",
    "gadepan_2": "Gadepan-II (Benfield)",
    "gadepan_3": "Gadepan-III (aMDEA)",
}

# Semantic colours
COLOR_BENCHMARK: str = "#95A5A6"    # Grey — benchmark / target line
COLOR_ANOMALY: str = "#E74C3C"      # Red — anomaly markers
COLOR_OPTIMAL: str = "#27AE60"      # Green — within-target / optimal
COLOR_WARNING: str = "#F39C12"      # Amber — warning band
COLOR_SAVINGS: str = "#2ECC71"      # Green — savings / improvement

# Qualitative palette for multi-model or multi-feature plots
MODEL_COLORS: dict[str, str] = {
    "xgboost": "#2E86C1",
    "lstm": "#E74C3C",
    "random_forest": "#27AE60",
    "linear_baseline": "#8E44AD",
}

# Extended palette for SHAP / feature importance
FEATURE_PALETTE: list[str] = [
    "#2E86C1", "#E74C3C", "#27AE60", "#F39C12", "#8E44AD",
    "#1ABC9C", "#D35400", "#34495E", "#16A085", "#C0392B",
]


# ---------------------------------------------------------------------------
# Theme Application
# ---------------------------------------------------------------------------

def apply_cfcl_theme(font_scale: float = 1.0) -> None:
    """Apply the CFCL corporate plotting theme globally.

    Sets ``matplotlib.rcParams`` for consistent styling across all figures
    produced in the pipeline. Call once at the start of any script or
    notebook that generates plots.

    Parameters
    ----------
    font_scale : float
        Multiplier for base font sizes (default 1.0).
    """
    base_size = 11 * font_scale

    plt.rcParams.update({
        # Figure
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "figure.facecolor": "#FAFAFA",
        "figure.edgecolor": "none",
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.facecolor": "#FFFFFF",

        # Fonts
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Segoe UI", "Helvetica", "Arial"],
        "font.size": base_size,

        # Axes
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#CCCCCC",
        "axes.linewidth": 0.8,
        "axes.titlesize": base_size * 1.3,
        "axes.titleweight": "bold",
        "axes.titlecolor": CFCL_DARK,
        "axes.labelsize": base_size * 1.05,
        "axes.labelcolor": CFCL_DARK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=FEATURE_PALETTE),

        # Grid
        "axes.grid": True,
        "grid.color": "#E0E0E0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.4,

        # Ticks
        "xtick.labelsize": base_size * 0.9,
        "ytick.labelsize": base_size * 0.9,
        "xtick.color": CFCL_DARK,
        "ytick.color": CFCL_DARK,
        "xtick.direction": "out",
        "ytick.direction": "out",

        # Legend
        "legend.fontsize": base_size * 0.9,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#CCCCCC",

        # Lines
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
    })


# ---------------------------------------------------------------------------
# Reusable Plot Helpers
# ---------------------------------------------------------------------------

def add_benchmark_line(
    ax: matplotlib.axes.Axes,
    benchmark: float,
    label: str = "Benchmark",
    **kwargs,
) -> None:
    """Add a horizontal benchmark / target line to an axes.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object.
    benchmark : float
        Y-value for the horizontal line.
    label : str
        Legend label.
    """
    defaults = dict(color=COLOR_BENCHMARK, linestyle="--", linewidth=1.5, alpha=0.8)
    defaults.update(kwargs)
    ax.axhline(y=benchmark, label=label, **defaults)


def add_savings_band(
    ax: matplotlib.axes.Axes,
    y_actual: np.ndarray,
    y_optimal: np.ndarray,
    x: Optional[np.ndarray] = None,
    label: str = "Savings potential",
) -> None:
    """Shade the area between actual and optimal SEC curves.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes.
    y_actual : array-like
        Actual SEC values.
    y_optimal : array-like
        Optimal (predicted lower-bound) SEC values.
    x : array-like, optional
        X-axis values. If ``None``, uses integer indices.
    label : str
        Legend label for the shaded band.
    """
    if x is None:
        x = np.arange(len(y_actual))
    ax.fill_between(
        x, y_actual, y_optimal,
        where=(np.asarray(y_actual) > np.asarray(y_optimal)),
        alpha=0.25,
        color=COLOR_SAVINGS,
        label=label,
        interpolate=True,
    )


def format_sec_axis(ax: matplotlib.axes.Axes, ylabel: bool = True) -> None:
    """Standardise the Y-axis for SEC plots.

    Formats ticks to 2 decimal places and adds the standard unit label.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes.
    ylabel : bool
        If ``True``, set the Y-axis label.
    """
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    if ylabel:
        ax.set_ylabel("SEC (Gcal/MT)")


def annotate_anomaly(
    ax: matplotlib.axes.Axes,
    x_pos: float,
    y_pos: float,
    label: str = "Anomaly",
) -> None:
    """Mark an anomaly point with a red diamond and optional annotation.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes.
    x_pos, y_pos : float
        Coordinates of the anomaly.
    label : str
        Text annotation.
    """
    ax.plot(x_pos, y_pos, marker="D", color=COLOR_ANOMALY, markersize=8,
            zorder=5, markeredgecolor="white", markeredgewidth=0.5)
    ax.annotate(
        label,
        xy=(x_pos, y_pos),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=8,
        color=COLOR_ANOMALY,
        arrowprops=dict(arrowstyle="->", color=COLOR_ANOMALY, lw=0.8),
    )


def save_figure(
    fig: matplotlib.figure.Figure,
    filename: str,
    output_dir: Optional[Path] = None,
    formats: tuple[str, ...] = ("png",),
) -> list[Path]:
    """Save a figure to the outputs directory.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure to save.
    filename : str
        Base filename (without extension).
    output_dir : Path, optional
        Target directory. Defaults to ``outputs/figures/``.
    formats : tuple of str
        File formats to save (e.g. ``("png", "svg")``).

    Returns
    -------
    list of Path
        Saved file paths.
    """
    project_root = Path(__file__).resolve().parents[2]
    target_dir = output_dir or (project_root / "outputs" / "figures")
    target_dir.mkdir(parents=True, exist_ok=True)

    # Strip any directory and existing extension if passed as Path or full filename
    clean_name = Path(filename).stem

    saved: list[Path] = []
    for fmt in formats:
        filepath = target_dir / f"{clean_name}.{fmt}"
        fig.savefig(filepath, format=fmt, bbox_inches="tight")
        saved.append(filepath)
    return saved


def create_model_comparison_bars(
    metrics: dict[str, dict[str, float]],
    metric_name: str = "R²",
    title: str = "Model Performance Comparison",
) -> matplotlib.figure.Figure:
    """Create a grouped bar chart comparing models on a given metric.

    Parameters
    ----------
    metrics : dict
        ``{model_name: {metric_name: value, ...}, ...}``
    metric_name : str
        Which metric to plot.
    title : str
        Plot title.

    Returns
    -------
    Figure
        The matplotlib Figure object.
    """
    models = list(metrics.keys())
    values = [metrics[m].get(metric_name, 0) for m in models]
    colors = [MODEL_COLORS.get(m, CFCL_LIGHT_BLUE) for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(models, values, color=colors, edgecolor="white", height=0.6)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=CFCL_DARK,
        )

    ax.set_xlabel(metric_name)
    ax.set_title(title)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.12)
    fig.tight_layout()
    return fig


def plant_color(plant_id: str) -> str:
    """Return the standard colour for a plant ID, with fallback."""
    return PLANT_COLORS.get(plant_id, CFCL_LIGHT_BLUE)


def plant_label(plant_id: str) -> str:
    """Return the display label for a plant ID."""
    return PLANT_LABELS.get(plant_id, plant_id)
