"""Tests for CFCL Telemetry Schemas, Range Contracts, and Boundary Validation.

Verifies:
1. Pydantic validation of DCS telemetry records.
2. Controllable vs uncontrollable feature partitioning.
3. Feature bounds definitions against plant_config.yaml.
4. Physical safety rejection of unfeasible process values.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.data.schema import (
    CATEGORICAL_COLUMNS,
    CONTROLLABLE_FEATURES,
    FEATURE_COLUMNS,
    RAW_COLUMNS,
    TARGET_COLUMN,
    UNCONTROLLABLE_FEATURES,
    PlantDataRecord,
    get_feature_bounds,
)
from src.api.schemas import PlantTelemetryInput


class TestSchemaDefinitions:
    """Verify schema column sets and engineering taxonomy."""

    def test_feature_column_counts(self) -> None:
        """Verify feature partitioning conforms to the 9-core process variable specification."""
        assert len(FEATURE_COLUMNS) == 9
        assert len(CONTROLLABLE_FEATURES) == 3
        assert len(UNCONTROLLABLE_FEATURES) == 5
        assert TARGET_COLUMN == "sec_gcal_per_mt"

    def test_controllable_variables(self) -> None:
        """Verify the 3 DCS manipulated variables (MVs)."""
        expected_mvs = {"steam_carbon_ratio", "reformer_outlet_temp", "loop_pressure"}
        assert set(CONTROLLABLE_FEATURES) == expected_mvs

    def test_uncontrollable_variables(self) -> None:
        """Verify the 5 float disturbance variables (DVs)."""
        expected_dvs = {
            "ambient_temp",
            "ambient_humidity",
            "cw_inlet_temp",
            "ng_calorific_value",
            "co2_slip_ppm",
        }
        assert set(UNCONTROLLABLE_FEATURES) == expected_dvs

    def test_raw_columns_dict_integrity(self) -> None:
        """Verify RAW_COLUMNS dictionary contains all operational tags and units."""
        for col_name in FEATURE_COLUMNS + [TARGET_COLUMN, "timestamp", "plant_id", "shift"]:
            assert col_name in RAW_COLUMNS
            spec = RAW_COLUMNS[col_name]
            assert hasattr(spec, "dtype")
            assert hasattr(spec, "unit")

    def test_get_feature_bounds(self) -> None:
        """Verify bounds dictionary returns valid min/max tuples."""
        bounds = get_feature_bounds()
        assert len(bounds) == 9
        for feat, (b_min, b_max) in bounds.items():
            assert b_min < b_max
            assert b_min >= 0.0


class TestPlantDataRecordValidation:
    """Verify Pydantic PlantDataRecord schema enforcement."""

    def test_valid_record(self) -> None:
        """Valid record should instantiate cleanly."""
        rec = PlantDataRecord(
            plant_id="gadepan_3",
            steam_carbon_ratio=3.05,
            reformer_outlet_temp=812.5,
            loop_pressure=158.0,
            ambient_temp=42.0,
            ambient_humidity=38.0,
            cw_inlet_temp=33.5,
            ng_calorific_value=8620.0,
            co2_slip_ppm=450.0,
            catalyst_age_days=180.0,
            shift="afternoon",
            sec_gcal_per_mt=4.92,
        )
        assert rec.plant_id == "gadepan_3"
        assert rec.steam_carbon_ratio == 3.05
        assert rec.sec_gcal_per_mt == 4.92

    def test_invalid_plant_id_rejected(self) -> None:
        """Unregistered plant unit must be rejected."""
        with pytest.raises(ValidationError):
            PlantDataRecord(
                plant_id="gadepan_99",  # Invalid
                steam_carbon_ratio=3.05,
                reformer_outlet_temp=812.5,
                loop_pressure=158.0,
                ambient_temp=42.0,
                ambient_humidity=38.0,
                cw_inlet_temp=33.5,
                ng_calorific_value=8620.0,
                co2_slip_ppm=450.0,
                catalyst_age_days=180.0,
                shift="afternoon",
                sec_gcal_per_mt=4.92,
            )

    def test_negative_pressure_rejected(self) -> None:
        """Physically impossible negative loop pressure must raise ValidationError."""
        with pytest.raises(ValidationError):
            PlantDataRecord(
                plant_id="gadepan_1",
                steam_carbon_ratio=3.05,
                reformer_outlet_temp=812.5,
                loop_pressure=-50.0,  # Impossible negative pressure
                ambient_temp=42.0,
                ambient_humidity=38.0,
                cw_inlet_temp=33.5,
                ng_calorific_value=8620.0,
                co2_slip_ppm=450.0,
                catalyst_age_days=180.0,
                shift="morning",
                sec_gcal_per_mt=5.15,
            )


class TestAPITelemetryInputValidation:
    """Verify FastAPI Pydantic V2 PlantTelemetryInput schema."""

    def test_valid_api_input(self, valid_telemetry_payload: dict) -> None:
        """API input should parse valid dictionary without error."""
        obj = PlantTelemetryInput(**valid_telemetry_payload)
        assert obj.plant_id == "gadepan_3"
        assert obj.steam_carbon_ratio == 3.05

    def test_coking_range_lower_bound(self, unsafe_telemetry_payload: dict) -> None:
        """S/C below 2.50 violates physical governor lower bound."""
        with pytest.raises(ValidationError):
            PlantTelemetryInput(**unsafe_telemetry_payload)
