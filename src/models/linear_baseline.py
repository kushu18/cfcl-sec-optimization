"""Linear Regression baseline benchmark for CFCL SEC Prediction.

Establishes a classical linear benchmark against which non-linear tree-based
and recurrent neural models are evaluated. Implements walk-forward cross-validation
with an operational gap to prevent temporal information leakage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.data.schema import TARGET_COLUMN
from src.utils.config_loader import get_model_params, get_paths, get_project_root, load_config
from src.utils.logger import get_logger


class LinearBaselineSEC:
    """Linear Regression baseline model with standard scaling and walk-forward CV."""

    def __init__(self, logger: Optional[Any] = None) -> None:
        self.logger = logger or get_logger("linear_baseline")
        self.model_cfg = load_config("model")
        self.params = get_model_params("linear_baseline")
        self.training_cfg = self.model_cfg.get("training", {})
        
        # Alpha regularization prevents ill-conditioned matrix inversion with 180+ features
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=1.0, fit_intercept=self.params.get("fit_intercept", True), random_state=42)
        
        self.feature_names: List[str] = []
        self.metrics: Dict[str, float] = {}
        self.cv_results: List[Dict[str, float]] = []

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Clean and extract feature matrix X and target y sorted chronologically."""
        df_clean = df.sort_values("timestamp").dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
        
        # Exclude non-feature columns
        exclude_cols = {
            "timestamp", "plant_id", "shift", TARGET_COLUMN,
            "catalyst_age_bucket", "outlier_features"
        }
        feature_cols = [c for c in df_clean.columns if c not in exclude_cols]
        self.feature_names = feature_cols
        
        # Fill any residual initial lag NaNs with column median/mean
        X = df_clean[feature_cols].copy()
        X = X.fillna(X.median())
        y = df_clean[TARGET_COLUMN].copy()
        
        return X, y

    def train_walk_forward(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Perform 5-fold walk-forward cross-validation with temporal gap."""
        wf_cfg = self.training_cfg.get("walk_forward", {})
        n_splits = wf_cfg.get("n_splits", 5)
        gap = wf_cfg.get("gap", 24)
        
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
        self.logger.info("Starting Walk-Forward CV: %d splits, %d-hour gap", n_splits, gap)
        
        cv_scores: List[Dict[str, float]] = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_va, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            fold_scaler = StandardScaler()
            X_tr_scaled = fold_scaler.fit_transform(X_tr)
            X_va_scaled = fold_scaler.transform(X_va)
            
            fold_model = Ridge(alpha=1.0, fit_intercept=self.params.get("fit_intercept", True), random_state=42)
            fold_model.fit(X_tr_scaled, y_tr)
            
            preds = fold_model.predict(X_va_scaled)
            
            r2 = float(r2_score(y_val, preds))
            mae = float(mean_absolute_error(y_val, preds))
            rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
            mape = float(mean_absolute_percentage_error(y_val, preds) * 100)
            
            cv_scores.append({
                "fold": fold + 1,
                "r2": round(r2, 4),
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "mape": round(mape, 4)
            })
            self.logger.info("  Fold %d: R²=%.4f | RMSE=%.4f | MAE=%.4f", fold + 1, r2, rmse, mae)
            
        self.cv_results = cv_scores
        
        # Aggregate validation metrics
        avg_r2 = float(np.mean([s["r2"] for s in cv_scores]))
        avg_mae = float(np.mean([s["mae"] for s in cv_scores]))
        avg_rmse = float(np.mean([s["rmse"] for s in cv_scores]))
        avg_mape = float(np.mean([s["mape"] for s in cv_scores]))
        
        return {
            "cv_r2": round(avg_r2, 4),
            "cv_mae": round(avg_mae, 4),
            "cv_rmse": round(avg_rmse, 4),
            "cv_mape": round(avg_mape, 4),
        }

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LinearBaselineSEC:
        """Fit scaler and linear model on full training dataset."""
        self.logger.info("Fitting Linear Baseline model on %d records (%d features)", len(X), X.shape[1])
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        preds = self.model.predict(X_scaled)
        self.metrics = {
            "train_r2": round(float(r2_score(y, preds)), 4),
            "train_mae": round(float(mean_absolute_error(y, preds)), 4),
            "train_rmse": round(float(np.sqrt(mean_squared_error(y, preds))), 4),
            "train_mape": round(float(mean_absolute_percentage_error(y, preds) * 100), 4),
        }
        self.logger.info("Training metrics: R²=%.4f | RMSE=%.4f | MAE=%.4f",
                         self.metrics["train_r2"], self.metrics["train_rmse"], self.metrics["train_mae"])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict specific energy consumption (Gcal/MT)."""
        X_clean = X[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X_clean)
        return self.model.predict(X_scaled)

    def save(self, filepath: Optional[Path | str] = None) -> Path:
        """Persist model, scaler, and training metadata."""
        if filepath is None:
            paths = get_paths()
            filepath = paths["models"] / "linear_baseline.joblib"
        else:
            filepath = Path(filepath)
            
        filepath.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "cv_results": self.cv_results,
            "params": self.params,
        }
        joblib.dump(artifact, filepath)
        self.logger.info("Saved Linear Baseline model artifact to %s", filepath)
        
        # Also save metrics JSON
        metrics_path = filepath.parent / "linear_baseline_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({"metrics": self.metrics, "cv_results": self.cv_results}, f, indent=2)
            
        return filepath

    def load(self, filepath: Path | str) -> LinearBaselineSEC:
        """Load persisted model artifact."""
        artifact = joblib.load(filepath)
        self.model = artifact["model"]
        self.scaler = artifact["scaler"]
        self.feature_names = artifact["feature_names"]
        self.metrics = artifact["metrics"]
        self.cv_results = artifact.get("cv_results", [])
        return self


if __name__ == "__main__":
    from src.utils.config_loader import get_project_root
    
    features_csv = get_project_root() / "data" / "processed" / "plant_features.csv"
    if not features_csv.exists():
        raise FileNotFoundError(f"Feature matrix not found: {features_csv}")
        
    df = pd.read_csv(features_csv)
    baseline = LinearBaselineSEC()
    X, y = baseline.prepare_data(df)
    
    # Run walk-forward CV
    cv_summary = baseline.train_walk_forward(X, y)
    print("\nWalk-Forward CV Summary:", cv_summary)
    
    # Train full model & save
    baseline.fit(X, y)
    baseline.save()
    print("\nLinear Baseline Training Complete. Model saved successfully.")
