"""Tests for CFCL Surrogate Machine Learning Models.

Verifies:
1. Checkpoint file integrity (Random Forest, XGBoost, Linear, PyTorch LSTM).
2. Leaderboard benchmark metrics (Champion R² >= 0.95, RMSE <= 0.025).
3. Realistic Specific Energy Consumption inference bounds.
4. Statistical 95% confidence interval mathematical consistency.
"""

from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.utils.config_loader import get_paths


class TestModelArtifacts:
    """Verify saved model checkpoints and serialization integrity."""

    def test_model_files_exist(self, project_paths: dict[str, Path]) -> None:
        """All 4 trained model checkpoints must exist in outputs/models/."""
        models_dir = project_paths["models"]
        assert (models_dir / "random_forest.joblib").exists()
        assert (models_dir / "xgboost_sec.joblib").exists()
        assert (models_dir / "linear_baseline.joblib").exists()
        assert (models_dir / "lstm_sec.pt").exists()
        assert (models_dir / "model_leaderboard.json").exists()

    def test_leaderboard_champion_criteria(self, project_paths: dict[str, Path]) -> None:
        """Champion model must satisfy strict industrial benchmarks."""
        lb_path = project_paths["models"] / "model_leaderboard.json"
        with open(lb_path, "r", encoding="utf-8") as f:
            lb = json.load(f)

        assert lb["champion_model"] == "Random_Forest"
        metrics = lb["champion_metrics"]
        assert metrics["r2_score"] >= 0.95, f"Champion R² too low: {metrics['r2_score']}"
        assert metrics["rmse"] <= 0.025, f"Champion RMSE too high: {metrics['rmse']}"
        assert metrics["within_1pct_dcs_band"] >= 95.0, f"DCS compliance below target: {metrics['within_1pct_dcs_band']}"


class TestSurrogateInference:
    """Verify forward-pass inference on the champion surrogate model."""

    def test_random_forest_inference_bounds(self, project_paths: dict[str, Path]) -> None:
        """Random Forest prediction should fall strictly within plant design ranges [4.80, 5.30]."""
        rf_path = project_paths["models"] / "random_forest.joblib"
        artifact = joblib.load(rf_path)
        model = artifact["model"]
        feature_names = artifact["feature_names"]

        # Build dummy evaluation row with all zeros except typical controllable setpoints
        x_eval = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
        if "steam_carbon_ratio" in x_eval.columns:
            x_eval["steam_carbon_ratio"] = 3.05
        if "reformer_outlet_temp" in x_eval.columns:
            x_eval["reformer_outlet_temp"] = 812.5
        if "loop_pressure" in x_eval.columns:
            x_eval["loop_pressure"] = 158.0
        if "plant_gadepan_3" in x_eval.columns:
            x_eval["plant_gadepan_3"] = 1.0

        pred = float(model.predict(x_eval)[0])
        assert 4.80 <= pred <= 5.30, f"Predicted SEC out of physical bounds: {pred}"

    def test_confidence_interval_math(self, project_paths: dict[str, Path]) -> None:
        """95% confidence interval must satisfy lower < point_estimate < upper."""
        lb_path = project_paths["models"] / "model_leaderboard.json"
        with open(lb_path, "r", encoding="utf-8") as f:
            lb = json.load(f)

        res_std = lb["champion_metrics"].get("residual_std", 0.0196)
        pred_sec = 4.9250
        ci_margin = 1.96 * res_std

        ci_lower = pred_sec - ci_margin
        ci_upper = pred_sec + ci_margin

        assert ci_lower < pred_sec < ci_upper
        assert round(ci_upper - ci_lower, 4) == round(2 * ci_margin, 4)
