"""Random Forest Regressor for CFCL SEC Prediction and Feature Importance Analysis.

Provides an ensemble decision tree baseline to validate non-linear process interactions
and provide a comparative cross-check against XGBoost feature importance rankings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from src.data.schema import TARGET_COLUMN
from src.utils.config_loader import get_model_params, get_paths, get_project_root, load_config
from src.utils.logger import get_logger


class RandomForestSEC:
    """Random Forest regressor model with walk-forward CV and feature importance extraction."""

    def __init__(self, logger: Optional[Any] = None) -> None:
        self.logger = logger or get_logger("random_forest")
        self.model_cfg = load_config("model")
        self.params = get_model_params("random_forest")
        self.training_cfg = self.model_cfg.get("training", {})
        
        self.model = RandomForestRegressor(
            n_estimators=self.params.get("n_estimators", 300),
            max_depth=self.params.get("max_depth", 10),
            min_samples_split=self.params.get("min_samples_split", 5),
            min_samples_leaf=self.params.get("min_samples_leaf", 2),
            max_features=self.params.get("max_features", "sqrt"),
            n_jobs=self.params.get("n_jobs", -1),
            random_state=42
        )
        
        self.feature_names: List[str] = []
        self.metrics: Dict[str, float] = {}
        self.cv_results: List[Dict[str, float]] = []
        self.feature_importances_: Dict[str, float] = {}

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Clean and extract feature matrix X and target y sorted chronologically."""
        df_clean = df.sort_values("timestamp").dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
        
        exclude_cols = {
            "timestamp", "plant_id", "shift", TARGET_COLUMN,
            "catalyst_age_bucket", "outlier_features"
        }
        feature_cols = [c for c in df_clean.columns if c not in exclude_cols]
        self.feature_names = feature_cols
        
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
        self.logger.info("Starting RF Walk-Forward CV: %d splits, %d-hour gap", n_splits, gap)
        
        cv_scores: List[Dict[str, float]] = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_va, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            fold_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=self.params.get("max_depth", 10),
                min_samples_split=self.params.get("min_samples_split", 5),
                min_samples_leaf=self.params.get("min_samples_leaf", 2),
                max_features=self.params.get("max_features", "sqrt"),
                n_jobs=-1,
                random_state=42
            )
            fold_model.fit(X_tr, y_tr)
            preds = fold_model.predict(X_va)
            
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

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RandomForestSEC:
        """Fit Random Forest model on full dataset."""
        self.logger.info("Fitting Random Forest model on %d records (%d features)", len(X), X.shape[1])
        self.model.fit(X, y)
        
        preds = self.model.predict(X)
        self.metrics = {
            "train_r2": round(float(r2_score(y, preds)), 4),
            "train_mae": round(float(mean_absolute_error(y, preds)), 4),
            "train_rmse": round(float(np.sqrt(mean_squared_error(y, preds))), 4),
            "train_mape": round(float(mean_absolute_percentage_error(y, preds) * 100), 4),
        }
        
        # Rank feature importances
        importances = self.model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        self.feature_importances_ = {
            self.feature_names[i]: round(float(importances[i]), 5) for i in sorted_idx
        }
        
        self.logger.info("RF Training metrics: R²=%.4f | RMSE=%.4f | MAE=%.4f",
                         self.metrics["train_r2"], self.metrics["train_rmse"], self.metrics["train_mae"])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict specific energy consumption (Gcal/MT)."""
        X_clean = X[self.feature_names].fillna(0)
        return self.model.predict(X_clean)

    def save(self, filepath: Optional[Path | str] = None) -> Path:
        """Persist model and metadata."""
        if filepath is None:
            paths = get_paths()
            filepath = paths["models"] / "random_forest.joblib"
        else:
            filepath = Path(filepath)
            
        filepath.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "model": self.model,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "cv_results": self.cv_results,
            "feature_importances": self.feature_importances_,
            "params": self.params,
        }
        joblib.dump(artifact, filepath)
        self.logger.info("Saved Random Forest model artifact to %s", filepath)
        
        metrics_path = filepath.parent / "random_forest_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({
                "metrics": self.metrics,
                "cv_results": self.cv_results,
                "top_10_features": dict(list(self.feature_importances_.items())[:10])
            }, f, indent=2)
            
        return filepath

    def load(self, filepath: Path | str) -> RandomForestSEC:
        """Load persisted model artifact."""
        artifact = joblib.load(filepath)
        self.model = artifact["model"]
        self.feature_names = artifact["feature_names"]
        self.metrics = artifact["metrics"]
        self.cv_results = artifact.get("cv_results", [])
        self.feature_importances_ = artifact.get("feature_importances", {})
        return self


if __name__ == "__main__":
    from src.utils.config_loader import get_project_root
    
    features_csv = get_project_root() / "data" / "processed" / "plant_features.csv"
    if not features_csv.exists():
        raise FileNotFoundError(f"Feature matrix not found: {features_csv}")
        
    df = pd.read_csv(features_csv)
    rf = RandomForestSEC()
    X, y = rf.prepare_data(df)
    
    cv_summary = rf.train_walk_forward(X, y)
    print("\nRF Walk-Forward CV Summary:", cv_summary)
    
    rf.fit(X, y)
    rf.save()
    print("\nRandom Forest Training Complete. Model saved successfully.")
    print("Top 5 Features:", list(rf.feature_importances_.items())[:5])
