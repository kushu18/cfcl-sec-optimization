"""XGBoost Regressor for CFCL Specific Energy Consumption (SEC) Optimization.

Primary production model leveraging gradient boosted decision trees to capture
complex non-linear interactions between operating variables, ambient conditions,
catalyst decay trajectories, and fleet efficiency dynamics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from src.data.schema import TARGET_COLUMN
from src.utils.config_loader import get_model_params, get_paths, get_project_root, load_config
from src.utils.logger import get_logger


class XGBoostSEC:
    """Production XGBoost model with walk-forward CV, early stopping, and metric tracking."""

    def __init__(self, logger: Optional[Any] = None) -> None:
        self.logger = logger or get_logger("xgboost_sec")
        self.model_cfg = load_config("model")
        self.params = get_model_params("xgboost")
        self.training_cfg = self.model_cfg.get("training", {})
        
        # XGBoost parameters from config
        self.early_stopping_rounds = self.params.get("early_stopping_rounds", 50)
        self.eval_metric = self.params.get("eval_metric", "rmse")
        
        self.model = xgb.XGBRegressor(
            n_estimators=self.params.get("n_estimators", 500),
            max_depth=self.params.get("max_depth", 6),
            learning_rate=self.params.get("learning_rate", 0.05),
            subsample=self.params.get("subsample", 0.8),
            colsample_bytree=self.params.get("colsample_bytree", 0.8),
            min_child_weight=self.params.get("min_child_weight", 3),
            reg_alpha=self.params.get("reg_alpha", 0.1),
            reg_lambda=self.params.get("reg_lambda", 1.0),
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=self.early_stopping_rounds,
            eval_metric=self.eval_metric,
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
        self.logger.info("Starting XGBoost Walk-Forward CV: %d splits, %d-hour gap", n_splits, gap)
        
        cv_scores: List[Dict[str, float]] = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_va, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            fold_model = xgb.XGBRegressor(
                n_estimators=self.params.get("n_estimators", 500),
                max_depth=self.params.get("max_depth", 6),
                learning_rate=self.params.get("learning_rate", 0.05),
                subsample=self.params.get("subsample", 0.8),
                colsample_bytree=self.params.get("colsample_bytree", 0.8),
                min_child_weight=self.params.get("min_child_weight", 3),
                reg_alpha=self.params.get("reg_alpha", 0.1),
                reg_lambda=self.params.get("reg_lambda", 1.0),
                random_state=42,
                n_jobs=-1,
                early_stopping_rounds=self.early_stopping_rounds,
                eval_metric=self.eval_metric,
            )
            
            fold_model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_val)],
                verbose=False
            )
            
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
                "mape": round(mape, 4),
                "best_iteration": int(fold_model.best_iteration) if hasattr(fold_model, "best_iteration") else -1
            })
            self.logger.info("  Fold %d: R²=%.4f | RMSE=%.4f | MAE=%.4f | Best Iter: %d",
                             fold + 1, r2, rmse, mae, cv_scores[-1]["best_iteration"])
            
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

    def fit(self, X: pd.DataFrame, y: pd.Series) -> XGBoostSEC:
        """Fit final model using train-val split for early stopping."""
        self.logger.info("Fitting production XGBoost model on %d records (%d features)", len(X), X.shape[1])
        
        # 80/20 temporal split for final model fit with early stopping
        val_size = int(len(X) * 0.2)
        X_train, y_train = X.iloc[:-val_size], y.iloc[:-val_size]
        X_val, y_val = X.iloc[-val_size:], y.iloc[-val_size:]
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Predict on holdout validation set
        val_preds = self.model.predict(X_val)
        self.metrics = {
            "val_r2": round(float(r2_score(y_val, val_preds)), 4),
            "val_mae": round(float(mean_absolute_error(y_val, val_preds)), 4),
            "val_rmse": round(float(np.sqrt(mean_squared_error(y_val, val_preds))), 4),
            "val_mape": round(float(mean_absolute_percentage_error(y_val, val_preds) * 100), 4),
            "best_iteration": int(self.model.best_iteration) if hasattr(self.model, "best_iteration") else -1
        }
        
        # Extract feature importances
        importances = self.model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        self.feature_importances_ = {
            self.feature_names[i]: round(float(importances[i]), 5) for i in sorted_idx
        }
        
        self.logger.info("XGBoost Validation metrics: R²=%.4f | RMSE=%.4f | MAE=%.4f",
                         self.metrics["val_r2"], self.metrics["val_rmse"], self.metrics["val_mae"])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict specific energy consumption (Gcal/MT)."""
        X_clean = X[self.feature_names].fillna(0)
        return self.model.predict(X_clean)

    def save(self, filepath: Optional[Path | str] = None) -> Path:
        """Persist model and metadata."""
        if filepath is None:
            paths = get_paths()
            filepath = paths["models"] / "xgboost_sec.joblib"
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
        self.logger.info("Saved XGBoost model artifact to %s", filepath)
        
        metrics_path = filepath.parent / "xgboost_sec_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({
                "metrics": self.metrics,
                "cv_results": self.cv_results,
                "top_15_features": dict(list(self.feature_importances_.items())[:15])
            }, f, indent=2)
            
        return filepath

    def load(self, filepath: Path | str) -> XGBoostSEC:
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
    xgb_model = XGBoostSEC()
    X, y = xgb_model.prepare_data(df)
    
    cv_summary = xgb_model.train_walk_forward(X, y)
    print("\nXGBoost Walk-Forward CV Summary:", cv_summary)
    
    xgb_model.fit(X, y)
    xgb_model.save()
    print("\nXGBoost Training Complete. Model saved successfully.")
    print("Top 5 Features:", list(xgb_model.feature_importances_.items())[:5])
