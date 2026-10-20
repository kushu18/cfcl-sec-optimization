"""Data schema contracts for the CFCL SEC Optimization pipeline.

Defines column specifications, valid ranges, data types, and validation
rules for raw DCS data, processed features, and API inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class PlantID(str, Enum):
    """Valid CFCL Gadepan plant identifiers."""
    GADEPAN_1 = "gadepan_1"
    GADEPAN_2 = "gadepan_2"
    GADEPAN_3 = "gadepan_3"


class ShiftType(str, Enum):
    """Operator shift classifications."""
    MORNING = "morning"      # 06:00 - 14:00
    AFTERNOON = "afternoon"  # 14:00 - 22:00
    NIGHT = "night"          # 22:00 - 06:00


@dataclass(frozen=True)
class ColumnSpec:
    """Specification for a single data column."""
    name: str
    display_name: str
    dtype: str
    unit: str
    min_value: float
    max_value: float
    description: str
    controllable: bool = False


# --- Raw DCS Column Specifications ---

RAW_COLUMNS: dict[str, ColumnSpec] = {
    "timestamp": ColumnSpec(
        name="timestamp",
        display_name="Timestamp",
        dtype="datetime64[ns]",
        unit="",
        min_value=0,
        max_value=0,
        description="Hourly DCS snapshot timestamp.",
    ),
    "plant_id": ColumnSpec(
        name="plant_id",
        display_name="Plant ID",
        dtype="category",
        unit="",
        min_value=0,
        max_value=0,
        description="CFCL Gadepan plant identifier.",
    ),
    "steam_carbon_ratio": ColumnSpec(
        name="steam_carbon_ratio",
        display_name="Steam-to-Carbon Ratio",
        dtype="float64",
        unit="mol/mol",
        min_value=2.8,
        max_value=3.4,
        description="Primary reformer steam injection ratio.",
        controllable=True,
    ),
    "loop_pressure": ColumnSpec(
        name="loop_pressure",
        display_name="Synthesis Loop Pressure",
        dtype="float64",
        unit="kg/cm²",
        min_value=145.0,
        max_value=185.0,
        description="Urea synthesis reactor operating pressure.",
        controllable=True,
    ),
    "reformer_outlet_temp": ColumnSpec(
        name="reformer_outlet_temp",
        display_name="Reformer Outlet Temperature",
        dtype="float64",
        unit="°C",
        min_value=780.0,
        max_value=830.0,
        description="Primary reformer exit gas temperature.",
        controllable=True,
    ),
    "cw_inlet_temp": ColumnSpec(
        name="cw_inlet_temp",
        display_name="Cooling Water Inlet Temperature",
        dtype="float64",
        unit="°C",
        min_value=24.0,
        max_value=40.0,
        description="Cooling tower return water temperature.",
        controllable=False,
    ),
    "ambient_temp": ColumnSpec(
        name="ambient_temp",
        display_name="Ambient Temperature",
        dtype="float64",
        unit="°C",
        min_value=12.0,
        max_value=48.0,
        description="Kota ambient dry-bulb temperature.",
        controllable=False,
    ),
    "ambient_humidity": ColumnSpec(
        name="ambient_humidity",
        display_name="Ambient Relative Humidity",
        dtype="float64",
        unit="%",
        min_value=10.0,
        max_value=100.0,
        description="Relative humidity — impacts cooling tower performance.",
        controllable=False,
    ),
    "co2_slip_ppm": ColumnSpec(
        name="co2_slip_ppm",
        display_name="CO₂ Slip",
        dtype="float64",
        unit="ppm",
        min_value=50.0,
        max_value=2500.0,
        description="CO₂ breakthrough from absorption column.",
        controllable=False,
    ),
    "ng_calorific_value": ColumnSpec(
        name="ng_calorific_value",
        display_name="Natural Gas Calorific Value",
        dtype="float64",
        unit="kcal/SCM",
        min_value=8400.0,
        max_value=9200.0,
        description="HBJ pipeline gas heating value.",
        controllable=False,
    ),
    "catalyst_age_days": ColumnSpec(
        name="catalyst_age_days",
        display_name="Catalyst Age",
        dtype="int64",
        unit="days",
        min_value=0,
        max_value=2500,
        description="Days since last catalyst replacement.",
        controllable=False,
    ),
    "shift": ColumnSpec(
        name="shift",
        display_name="Operator Shift",
        dtype="category",
        unit="",
        min_value=0,
        max_value=0,
        description="Current operator shift classification.",
        controllable=False,
    ),
    "sec_gcal_per_mt": ColumnSpec(
        name="sec_gcal_per_mt",
        display_name="Specific Energy Consumption",
        dtype="float64",
        unit="Gcal/MT",
        min_value=4.5,
        max_value=6.0,
        description="Total energy consumed per metric ton of urea produced.",
        controllable=False,
    ),
}


# Feature columns (numeric inputs to the model)
FEATURE_COLUMNS: list[str] = [
    "steam_carbon_ratio",
    "loop_pressure",
    "reformer_outlet_temp",
    "cw_inlet_temp",
    "ambient_temp",
    "ambient_humidity",
    "co2_slip_ppm",
    "ng_calorific_value",
    "catalyst_age_days",
]

TARGET_COLUMN: str = "sec_gcal_per_mt"

CATEGORICAL_COLUMNS: list[str] = ["plant_id", "shift"]

CONTROLLABLE_FEATURES: list[str] = [
    col for col, spec in RAW_COLUMNS.items() if spec.controllable
]

UNCONTROLLABLE_FEATURES: list[str] = [
    col for col, spec in RAW_COLUMNS.items()
    if not spec.controllable and spec.dtype == "float64" and col != TARGET_COLUMN
]


# --- Pydantic Validation Models ---

class PlantDataRecord(BaseModel):
    """Validates a single row of raw DCS plant data."""

    plant_id: PlantID
    steam_carbon_ratio: float = Field(ge=2.8, le=3.4)
    loop_pressure: float = Field(ge=145.0, le=185.0)
    reformer_outlet_temp: float = Field(ge=780.0, le=830.0)
    cw_inlet_temp: float = Field(ge=24.0, le=40.0)
    ambient_temp: float = Field(ge=12.0, le=48.0)
    ambient_humidity: float = Field(ge=10.0, le=100.0)
    co2_slip_ppm: float = Field(ge=50.0, le=2500.0)
    ng_calorific_value: float = Field(ge=8400.0, le=9200.0)
    catalyst_age_days: int = Field(ge=0, le=2500)
    shift: ShiftType
    sec_gcal_per_mt: Optional[float] = Field(default=None, ge=4.5, le=6.0)

    model_config = {"str_strip_whitespace": True}


class PredictionRequest(BaseModel):
    """API request schema for SEC prediction."""

    plant: PlantID
    steam_carbon_ratio: float = Field(ge=2.8, le=3.4, description="Steam-to-carbon molar ratio")
    loop_pressure: float = Field(ge=145.0, le=185.0, description="Synthesis loop pressure in kg/cm²")
    reformer_outlet_temp: float = Field(ge=780.0, le=830.0, description="Reformer outlet temp in °C")
    cw_inlet_temp: float = Field(ge=24.0, le=40.0, description="Cooling water inlet temp in °C")
    ambient_temp: float = Field(ge=12.0, le=48.0, description="Ambient temperature in °C")
    ambient_humidity: float = Field(ge=10.0, le=100.0, description="Relative humidity in %")
    co2_slip_ppm: float = Field(ge=50.0, le=2500.0, description="CO₂ slip in ppm")
    ng_calorific_value: float = Field(ge=8400.0, le=9200.0, description="NG calorific value in kcal/SCM")
    catalyst_age_days: int = Field(ge=0, le=2500, description="Days since catalyst replacement")
    shift: ShiftType = Field(description="Current operator shift")


class SECDriver(BaseModel):
    """A single feature's contribution to SEC deviation."""
    feature: str
    impact: str
    actionable: bool


class PredictionResponse(BaseModel):
    """API response schema for SEC prediction."""
    predicted_sec_gcal_mt: float
    optimal_sec_gcal_mt: float
    savings_potential_inr_per_day: float
    top_3_drivers: list[SECDriver]
    recommendation: str


class HealthResponse(BaseModel):
    """API health check response."""
    status: str
    model_loaded: bool
    version: str


class DriftReport(BaseModel):
    """Data drift detection report."""
    timestamp: str
    features_drifted: list[str]
    severity: str  # LOW, MODERATE, HIGH
    recommendation: str
    model_still_reliable: bool


def get_feature_bounds() -> dict[str, tuple[float, float]]:
    """Return safety bounds for all numeric features.

    Returns:
        Dictionary mapping feature name to (min, max) tuple.
    """
    return {
        col: (spec.min_value, spec.max_value)
        for col, spec in RAW_COLUMNS.items()
        if spec.dtype in ("float64", "int64") and col != TARGET_COLUMN
    }


def get_controllable_bounds() -> dict[str, tuple[float, float]]:
    """Return safety bounds for operator-controllable features only.

    These are the features that the constrained optimizer is allowed to adjust.

    Returns:
        Dictionary mapping feature name to (min, max) tuple.
    """
    return {
        col: (spec.min_value, spec.max_value)
        for col, spec in RAW_COLUMNS.items()
        if spec.controllable
    }
