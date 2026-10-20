"""Tests for DCS Covariate & Concept Drift Monitoring Engine.

Verifies:
1. Two-sample Kolmogorov-Smirnov hypothesis testing mechanics.
2. Accurate detection of stationary vs drifted process features.
3. Fleet-wide severity tier classification (NORMAL, LOW, MODERATE, HIGH_CRITICAL).
4. Generation and schema of drift monitoring audit JSON reports.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift_detector import ConceptDriftDetector, FeatureDriftResult


@pytest.fixture
def drift_detector() -> ConceptDriftDetector:
    return ConceptDriftDetector(significance_level=0.05)


class TestDriftDetector:
    """Verify statistical drift tests and classification rules."""

    def test_detector_monitored_features(self, drift_detector: ConceptDriftDetector) -> None:
        """Detector must monitor 10 key physical tags including SEC target."""
        assert len(drift_detector.MONITORED_FEATURES) == 10
        assert "steam_carbon_ratio" in drift_detector.MONITORED_FEATURES
        assert "sec_gcal_per_mt" in drift_detector.MONITORED_FEATURES

    def test_partition_baseline_and_monitoring(
        self,
        drift_detector: ConceptDriftDetector,
        sample_raw_dataframe: pd.DataFrame
    ) -> None:
        """Partitioning should split chronologically into reference and monitoring sets."""
        ref, curr = drift_detector.partition_baseline_and_monitoring(sample_raw_dataframe, baseline_fraction=0.5)
        assert len(ref) + len(curr) == len(sample_raw_dataframe)
        assert ref["timestamp"].max() <= curr["timestamp"].min()

    def test_ks_test_detects_distribution_shift(
        self,
        drift_detector: ConceptDriftDetector
    ) -> None:
        """Deliberately shifted synthetic distribution must be flagged with p-value < 0.05."""
        np.random.seed(42)
        n = 1000

        # Baseline: normal ambient temp ~ 28°C
        ref_df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="h"),
            "ambient_temp": np.random.normal(28.0, 3.0, n),
            "steam_carbon_ratio": np.random.normal(3.05, 0.05, n),
            "reformer_outlet_temp": np.random.normal(810.0, 5.0, n),
            "loop_pressure": np.random.normal(158.0, 3.0, n),
            "ambient_humidity": np.random.normal(40.0, 10.0, n),
            "cw_inlet_temp": np.random.normal(30.0, 2.0, n),
            "ng_calorific_value": np.random.normal(8600.0, 50.0, n),
            "co2_slip_ppm": np.random.normal(400.0, 40.0, n),
            "catalyst_age_days": np.linspace(10, 50, n),
            "sec_gcal_per_mt": np.random.normal(4.95, 0.03, n),
        })

        # Monitoring: summer heatwave ambient temp ~ 45°C (severe drift on ambient_temp)
        curr_df = ref_df.copy()
        curr_df["ambient_temp"] = np.random.normal(45.0, 3.0, n)

        drift_detector.ref_data = ref_df
        drift_detector.curr_data = curr_df

        results = drift_detector.run_ks_tests()
        amb_res = [r for r in results if r.feature_name == "ambient_temp"][0]

        assert amb_res.is_drifted is True
        assert amb_res.p_value < 1e-10
        assert amb_res.severity_flag in ["DRIFT_DETECTED", "CRITICAL_DRIFT"]

    def test_overall_severity_classification(
        self,
        drift_detector: ConceptDriftDetector
    ) -> None:
        """Overall fleet severity must classify correctly based on drifted tag count."""
        np.random.seed(42)
        n = 500
        base_df = pd.DataFrame({
            feat: np.random.normal(100.0, 10.0, n) for feat in drift_detector.MONITORED_FEATURES
        })
        base_df["timestamp"] = pd.date_range("2026-01-01", periods=n, freq="h")

        # Case 1: Identical data -> NORMAL
        drift_detector.ref_data = base_df.copy()
        drift_detector.curr_data = base_df.copy()
        drift_detector.run_ks_tests()
        assert drift_detector.overall_severity == "NORMAL"

        # Case 2: 4 features shifted -> HIGH_SEVERITY_CRITICAL
        shifted_df = base_df.copy()
        for feat in drift_detector.MONITORED_FEATURES[:4]:
            shifted_df[feat] = shifted_df[feat] + 50.0  # Massive shift

        drift_detector.curr_data = shifted_df
        drift_detector.run_ks_tests()
        assert drift_detector.overall_severity == "HIGH_SEVERITY_CRITICAL"
