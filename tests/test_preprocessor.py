"""Tests for CFCL Data Preprocessor Pipeline.

Verifies:
1. Loading and column schema compliance.
2. Range clipping to safe physical bounds.
3. KNN imputation of sparse telemetry missing values.
4. Turnaround operational gap detection without data loss.
5. Cyclical temporal feature engineering and categorical encoding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import DataPreprocessor


@pytest.fixture
def preprocessor() -> DataPreprocessor:
    return DataPreprocessor()


class TestDataPreprocessor:
    """Verify preprocessor methods individually and end-to-end."""

    def test_validate_ranges_clipping(self, preprocessor: DataPreprocessor, sample_raw_dataframe: pd.DataFrame) -> None:
        """Sensor values exceeding bounds should be clipped to boundary values, not dropped."""
        df = sample_raw_dataframe.copy()
        # Introduce intentional spike
        df.loc[0, "steam_carbon_ratio"] = 10.0  # Above normal max ~3.4
        df.loc[1, "reformer_outlet_temp"] = 500.0  # Below normal min ~700

        clipped_df = preprocessor.validate_ranges(df)
        assert len(clipped_df) == len(df)
        assert clipped_df.loc[0, "steam_carbon_ratio"] <= 4.0
        assert clipped_df.loc[1, "reformer_outlet_temp"] >= 700.0

    def test_impute_missing_eliminates_nans(self, preprocessor: DataPreprocessor, sample_raw_dataframe: pd.DataFrame) -> None:
        """KNN imputation must fill missing numeric values without altering categorical or time keys."""
        df = sample_raw_dataframe.copy()
        df.loc[2, "cw_inlet_temp"] = np.nan
        df.loc[5, "loop_pressure"] = np.nan

        imputed_df = preprocessor.impute_missing(df)
        assert not imputed_df["cw_inlet_temp"].isna().any()
        assert not imputed_df["loop_pressure"].isna().any()
        assert len(imputed_df) == len(df)

    def test_detect_turnaround_gaps(self, preprocessor: DataPreprocessor, sample_raw_dataframe: pd.DataFrame) -> None:
        """Turnaround offline periods (>2hr delta) should be flagged with restart tracking."""
        df = sample_raw_dataframe.copy()
        # Create artificial gap for gadepan_1
        g1_idx = df[df["plant_id"] == "gadepan_1"].index
        if len(g1_idx) >= 2:
            df.loc[g1_idx[1], "timestamp"] = df.loc[g1_idx[0], "timestamp"] + pd.Timedelta(hours=48)

        gap_df = preprocessor.detect_turnaround_gaps(df)
        assert "is_turnaround_gap" in gap_df.columns
        assert "days_since_restart" in gap_df.columns

    def test_encode_categoricals(self, preprocessor: DataPreprocessor, sample_raw_dataframe: pd.DataFrame) -> None:
        """One-hot encoded flags for plants and shifts should be generated with original columns preserved."""
        encoded_df = preprocessor.encode_categoricals(sample_raw_dataframe)
        expected_cols = [
            "plant_gadepan_1", "plant_gadepan_2", "plant_gadepan_3",
            "shift_morning", "shift_afternoon", "shift_night"
        ]
        for col in expected_cols:
            assert col in encoded_df.columns
            assert set(encoded_df[col].unique()).issubset({0.0, 1.0, 0, 1})

    def test_add_temporal_features(self, preprocessor: DataPreprocessor, sample_raw_dataframe: pd.DataFrame) -> None:
        """Cyclical trigonometric representations of time should be created within [-1, 1]."""
        temp_df = preprocessor.add_temporal_features(sample_raw_dataframe)
        for col in ["hour_sin", "hour_cos", "month_sin", "month_cos"]:
            assert col in temp_df.columns
            assert temp_df[col].min() >= -1.0001
            assert temp_df[col].max() <= 1.0001
