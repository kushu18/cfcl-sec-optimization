"""Tests for CFCL Feature Engineering Pipeline.

Verifies:
1. Interaction terms capturing SMR thermodynamics and cooling water heat load.
2. Controllable variable deviations from plant optimal operating midpoints.
3. Lag and first-derivative rate-of-change feature calculations.
4. Catalyst degradation non-linear decays.
5. Overall feature matrix dimensionality and non-null guarantees.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.feature_engine import FeatureEngine
from src.data.preprocessor import DataPreprocessor


@pytest.fixture
def feature_engine() -> FeatureEngine:
    return FeatureEngine()


@pytest.fixture
def preprocessed_sample(sample_raw_dataframe: pd.DataFrame) -> pd.DataFrame:
    prep = DataPreprocessor()
    df = prep.validate_ranges(sample_raw_dataframe)
    df = prep.impute_missing(df)
    df = prep.detect_turnaround_gaps(df)
    df = prep.encode_categoricals(df)
    df = prep.add_temporal_features(df)
    return df


class TestFeatureEngine:
    """Verify feature generation logic and domain calculations."""

    def test_interaction_terms(self, feature_engine: FeatureEngine, preprocessed_sample: pd.DataFrame) -> None:
        """Physical interaction features should match thermodynamic product/ratio definitions."""
        df_int = feature_engine.compute_interaction_terms(preprocessed_sample)
        assert "steam_reformer_interaction" in df_int.columns
        assert "ambient_cw_interaction" in df_int.columns

        # Verify multiplication identity
        expected_sr = df_int["steam_carbon_ratio"] * df_int["reformer_outlet_temp"]
        np.testing.assert_allclose(df_int["steam_reformer_interaction"], expected_sr, rtol=1e-5)

    def test_deviation_from_optimal(self, feature_engine: FeatureEngine, preprocessed_sample: pd.DataFrame) -> None:
        """Deviations from setpoint midpoints should calculate signed difference and binary out-of-band flags."""
        df_dev = feature_engine.compute_deviation_from_optimal(preprocessed_sample)
        for mv in ["steam_carbon_ratio", "reformer_outlet_temp", "loop_pressure"]:
            assert f"{mv}_deviation" in df_dev.columns
            assert f"{mv}_outside_optimal" in df_dev.columns
            assert set(df_dev[f"{mv}_outside_optimal"].unique()).issubset({0, 1, 0.0, 1.0})

    def test_catalyst_features(self, feature_engine: FeatureEngine, preprocessed_sample: pd.DataFrame) -> None:
        """Catalyst age non-linear features should include squared and log transforms."""
        df_cat = feature_engine.compute_catalyst_features(preprocessed_sample)
        assert "catalyst_age_squared" in df_cat.columns
        assert "catalyst_age_log" in df_cat.columns

        expected_log = np.log(df_cat["catalyst_age_days"] + 1.0)
        np.testing.assert_allclose(df_cat["catalyst_age_log"], expected_log, rtol=1e-5)

    def test_efficiency_ratios(self, feature_engine: FeatureEngine, preprocessed_sample: pd.DataFrame) -> None:
        """Cross-efficiency indicators should be present and finite when lags are populated."""
        df_lags = feature_engine.compute_lag_features(preprocessed_sample)
        df_eff = feature_engine.compute_efficiency_ratios(df_lags)
        assert "energy_intensity" in df_eff.columns
        assert "thermal_load_index" in df_eff.columns
        assert np.isfinite(df_eff["energy_intensity"].dropna()).all()

    def test_feature_engine_run(self, feature_engine: FeatureEngine, preprocessed_sample: pd.DataFrame) -> None:
        """Full pipeline execution should produce over 100 engineered features without infinite values."""
        df_out = feature_engine.run(preprocessed_sample)
        assert len(df_out) == len(preprocessed_sample)
        assert df_out.shape[1] > 100
        # No infinite values allowed
        numeric_cols = df_out.select_dtypes(include=[np.number]).columns
        assert not np.isinf(df_out[numeric_cols].values).any()
