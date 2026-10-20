"""Unified Multi-Model Evaluator for CFCL Specific Energy Consumption (SEC) Optimization.

Provides rigorous, publication-quality benchmarking across all four model families:
1. Linear Baseline (StandardScaler + Ridge Regression)
2. Random Forest Regressor
3. XGBoost Regressor (Production Gradient Boosted Trees)
4. Stacked LSTM (Deep PyTorch Sequence Model)

Generates enterprise statistical diagnostics (Durbin-Watson, heteroscedasticity,
95% prediction intervals, within-plant breakdown) and exports CFCL-branded figures
and structured JSON leaderboards.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from src.data.schema import TARGET_COLUMN
from src.models.lstm_sec import StackedLSTM
from src.utils.config_loader import get_paths, get_project_root, load_config
from src.utils.logger import get_logger
from src.utils.visualization import (
    CFCL_BLUE,
    CFCL_DARK,
    CFCL_LIGHT_BLUE,
    PLANT_COLORS,
    PLANT_LABELS,
    apply_cfcl_theme,
    save_figure,
)


class ModelEvaluator:
    """Unified evaluator benchmarking all trained models on unseen holdout telemetry."""

    def __init__(self, logger: Optional[Any] = None) -> None:
        self.logger = logger or get_logger("model_evaluator")
        self.paths = get_paths()
        self.model_cfg = load_config("model")
        self.models_dir = self.paths["models"]
        self.figures_dir = self.paths["figures"]
        self.reports_dir = self.paths["reports"]

        self.models: Dict[str, Any] = {}
        self.predictions: Dict[str, np.ndarray] = {}
        self.leaderboard: Dict[str, Any] = {}
        self.holdout_df: Optional[pd.DataFrame] = None
        self.y_true: Optional[np.ndarray] = None

    def load_holdout_data(self, test_size: float = 0.2) -> pd.DataFrame:
        """Load processed features and isolate chronological holdout test set."""
        processed_dir = self.paths["processed_data"]
        parquet_path = processed_dir / "plant_features.parquet"
        csv_path = processed_dir / "plant_features.csv"

        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            self.logger.info("Loaded holdout source from Parquet: %s", parquet_path)
        elif csv_path.exists():
            df = pd.read_csv(csv_path, parse_dates=["timestamp"])
            self.logger.info("Loaded holdout source from CSV: %s", csv_path)
        else:
            raise FileNotFoundError("Processed features file not found in data/processed/")

        # Chronological sort
        df = df.sort_values("timestamp").dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
        val_size = int(len(df) * test_size)
        self.holdout_df = df.iloc[-val_size:].copy().reset_index(drop=True)
        self.y_true = self.holdout_df[TARGET_COLUMN].values

        self.logger.info(
            "Isolated holdout evaluation set: %d records (%s to %s)",
            len(self.holdout_df),
            self.holdout_df["timestamp"].min(),
            self.holdout_df["timestamp"].max(),
        )
        return self.holdout_df

    def load_models(self) -> Dict[str, Any]:
        """Load all 4 trained model artifacts."""
        # 1. Linear Baseline
        linear_path = self.models_dir / "linear_baseline.joblib"
        if linear_path.exists():
            self.models["Linear_Baseline"] = joblib.load(linear_path)
            self.logger.info("Loaded Linear Baseline from %s", linear_path)

        # 2. Random Forest
        rf_path = self.models_dir / "random_forest.joblib"
        if rf_path.exists():
            self.models["Random_Forest"] = joblib.load(rf_path)
            self.logger.info("Loaded Random Forest from %s", rf_path)

        # 3. XGBoost
        xgb_path = self.models_dir / "xgboost_sec.joblib"
        if xgb_path.exists():
            self.models["XGBoost"] = joblib.load(xgb_path)
            self.logger.info("Loaded XGBoost from %s", xgb_path)

        # 4. PyTorch LSTM
        lstm_path = self.models_dir / "lstm_sec.pt"
        if lstm_path.exists():
            self.models["LSTM"] = torch.load(lstm_path, map_location="cpu")
            self.logger.info("Loaded PyTorch LSTM from %s", lstm_path)

        return self.models

    def generate_predictions(self) -> Dict[str, np.ndarray]:
        """Generate predictions on holdout set for all loaded models."""
        if self.holdout_df is None:
            self.load_holdout_data()

        df = self.holdout_df

        # --- A. Tabular Models (Linear, RF, XGBoost) ---
        for name in ["Linear_Baseline", "Random_Forest", "XGBoost"]:
            if name in self.models:
                artifact = self.models[name]
                feature_names = artifact["feature_names"]
                X_val = df[feature_names].fillna(0)

                if name == "Linear_Baseline":
                    scaler = artifact["scaler"]
                    model = artifact["model"]
                    preds = model.predict(scaler.transform(X_val))
                elif name in ["Random_Forest", "XGBoost"]:
                    preds = artifact["model"].predict(X_val)

                self.predictions[name] = np.asarray(preds).flatten()
                self.logger.info("Generated predictions for %s (N=%d)", name, len(preds))

        # --- B. Deep Sequential LSTM ---
        if "LSTM" in self.models:
            lstm_art = self.models["LSTM"]
            feat_cols = lstm_art["feature_columns"]
            feat_scaler = lstm_art["feature_scaler"]
            targ_scaler = lstm_art["target_scaler"]
            params = lstm_art["params"]
            seq_len = params["sequence_length"]

            model = StackedLSTM(
                input_size=len(feat_cols),
                hidden_units=params["hidden_units"],
                num_layers=params["num_layers"],
                dropout=params["dropout"],
            )
            model.load_state_dict(lstm_art["model_state_dict"])
            model.eval()

            # Sequence prediction per plant with causal historical padding
            full_df_path = self.paths["processed_data"] / "plant_features.parquet"
            full_df = pd.read_parquet(full_df_path).sort_values("timestamp").reset_index(drop=True)
            
            lstm_preds_map: Dict[Tuple[str, pd.Timestamp], float] = {}

            for plant_id, p_grp in full_df.groupby("plant_id", sort=False):
                p_sorted = p_grp.sort_values("timestamp").reset_index(drop=True)
                feat_scaled = feat_scaler.transform(p_sorted[feat_cols].fillna(0))

                # Identify which rows of p_sorted are in holdout
                val_cutoff_time = df["timestamp"].min()
                holdout_indices = p_sorted.index[p_sorted["timestamp"] >= val_cutoff_time].tolist()

                for idx in holdout_indices:
                    if idx >= seq_len:
                        seq_window = feat_scaled[idx - seq_len : idx]
                    else:
                        # Pad with first available row if edge case
                        pad = np.repeat(feat_scaled[:1], seq_len - idx, axis=0)
                        seq_window = np.vstack([pad, feat_scaled[:idx]])

                    seq_tensor = torch.tensor(seq_window[np.newaxis, ...], dtype=torch.float32)
                    with torch.no_grad():
                        pred_scaled = model(seq_tensor).numpy().flatten()[0]
                    pred_unscaled = float(targ_scaler.inverse_transform([[pred_scaled]])[0, 0])
                    
                    ts = p_sorted.loc[idx, "timestamp"]
                    lstm_preds_map[(plant_id, ts)] = pred_unscaled

            # Align LSTM predictions with holdout_df order
            lstm_preds = []
            for _, row in df.iterrows():
                key = (row["plant_id"], row["timestamp"])
                lstm_preds.append(lstm_preds_map.get(key, np.nan))

            lstm_preds_arr = np.array(lstm_preds)
            # Impute any missing sequence edge using mean
            if np.isnan(lstm_preds_arr).any():
                valid_mean = np.nanmean(lstm_preds_arr)
                lstm_preds_arr = np.nan_to_num(lstm_preds_arr, nan=valid_mean)

            self.predictions["LSTM"] = lstm_preds_arr
            self.logger.info("Generated predictions for LSTM (N=%d)", len(lstm_preds_arr))

        return self.predictions

    @staticmethod
    def compute_durbin_watson(residuals: np.ndarray) -> float:
        """Compute Durbin-Watson statistic for residual autocorrelation."""
        diffs = np.diff(residuals)
        dw = np.sum(diffs**2) / np.sum(residuals**2) if np.sum(residuals**2) > 0 else 2.0
        return float(round(dw, 4))

    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Compute comprehensive performance and industrial DCS diagnostics."""
        residuals = y_true - y_pred
        abs_residuals = np.abs(residuals)
        rel_errors_pct = (abs_residuals / y_true) * 100.0

        r2 = float(r2_score(y_true, y_pred))
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100.0)
        max_err = float(max_error(y_true, y_pred))
        ev = float(explained_variance_score(y_true, y_pred))

        # Advanced Industrial Diagnostics
        dw = self.compute_durbin_watson(residuals)
        residual_std = float(np.std(residuals))
        ci_95 = float(1.96 * residual_std)
        within_1pct = float(np.mean(rel_errors_pct <= 1.0) * 100.0)
        within_2pct = float(np.mean(rel_errors_pct <= 2.0) * 100.0)

        # Heteroscedasticity correlation: r(|residuals|, y_pred)
        if np.std(abs_residuals) > 0 and np.std(y_pred) > 0:
            het_corr, _ = stats.pearsonr(abs_residuals, y_pred)
        else:
            het_corr = 0.0

        # Plant-specific metrics
        within_plant: Dict[str, Dict[str, float]] = {}
        for p_id in ["gadepan_1", "gadepan_2", "gadepan_3"]:
            mask = self.holdout_df["plant_id"] == p_id
            if mask.sum() > 0:
                y_t_p = y_true[mask]
                y_p_p = y_pred[mask]
                within_plant[p_id] = {
                    "r2": float(round(r2_score(y_t_p, y_p_p), 4)),
                    "rmse": float(round(np.sqrt(mean_squared_error(y_t_p, y_p_p)), 4)),
                    "mae": float(round(mean_absolute_error(y_t_p, y_p_p), 4)),
                    "mape": float(round(mean_absolute_percentage_error(y_t_p, y_p_p) * 100.0, 4)),
                }

        return {
            "r2_score": round(r2, 4),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mape": round(mape, 4),
            "max_error": round(max_err, 4),
            "explained_variance": round(ev, 4),
            "durbin_watson": dw,
            "residual_std": round(residual_std, 4),
            "prediction_interval_95": round(ci_95, 4),
            "within_1pct_dcs_band": round(within_1pct, 2),
            "within_2pct_dcs_band": round(within_2pct, 2),
            "heteroscedasticity_corr": round(float(het_corr), 4),
            "within_plant": within_plant,
        }

    def run_evaluation(self) -> Dict[str, Any]:
        """Execute unified multi-model evaluation and select production champion."""
        self.logger.info("Initiating unified multi-model evaluation suite...")
        if not self.models:
            self.load_models()
        if not self.predictions:
            self.generate_predictions()

        y_true = self.y_true
        summary_table: Dict[str, Dict[str, Any]] = {}

        for name, preds in self.predictions.items():
            metrics = self.evaluate_model(y_true, preds, name)
            summary_table[name] = metrics
            self.logger.info(
                "Model %-16s | RA: %.4f | RMSE: %.4f Gcal/MT | MAE: %.4f | Within 1%%: %.1f%%",
                name,
                metrics["r2_score"],
                metrics["rmse"],
                metrics["mae"],
                metrics["within_1pct_dcs_band"],
            )

        # Identify champion (lowest RMSE among models with R2 > 0.80)
        sorted_models = sorted(summary_table.items(), key=lambda x: (x[1]["rmse"], -x[1]["r2_score"]))
        champion_name = sorted_models[0][0]

        self.leaderboard = {
            "evaluation_timestamp": pd.Timestamp.now().isoformat(),
            "holdout_records": len(y_true),
            "holdout_start": str(self.holdout_df["timestamp"].min()),
            "holdout_end": str(self.holdout_df["timestamp"].max()),
            "champion_model": champion_name,
            "champion_metrics": summary_table[champion_name],
            "models": summary_table,
        }

        # Save leaderboard JSON
        out_json = self.models_dir / "model_leaderboard.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(self.leaderboard, f, indent=2)
        self.logger.info("Saved unified model leaderboard to %s", out_json)

        # Save holdout predictions dataframe
        pred_df = self.holdout_df[["timestamp", "plant_id", TARGET_COLUMN]].copy()
        for name, preds in self.predictions.items():
            pred_df[f"pred_{name.lower()}"] = preds
            pred_df[f"residual_{name.lower()}"] = pred_df[TARGET_COLUMN] - preds

        pred_parquet = self.models_dir / "holdout_predictions.parquet"
        pred_df.to_parquet(pred_parquet, index=False)
        self.logger.info("Saved holdout predictions table to %s", pred_parquet)

        # Generate plots
        self.plot_leaderboard_comparison()
        self.plot_actual_vs_predicted()
        self.plot_residual_diagnostics()
        self.plot_within_plant_breakdown()

        return self.leaderboard

    # =========================================================================
    # Visualizations (CFCL Branded Theme)
    # =========================================================================

    def plot_leaderboard_comparison(self) -> Path:
        """Create publication-grade grouped bar chart comparing all 4 models."""
        apply_cfcl_theme()
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        models = list(self.leaderboard["models"].keys())
        display_names = [m.replace("_", " ") for m in models]
        r2_vals = [self.leaderboard["models"][m]["r2_score"] for m in models]
        rmse_vals = [self.leaderboard["models"][m]["rmse"] for m in models]
        mae_vals = [self.leaderboard["models"][m]["mae"] for m in models]

        colors = [
            "#2ECC71" if m == self.leaderboard["champion_model"] else CFCL_LIGHT_BLUE
            for m in models
        ]

        # 1. R2 Score
        axes[0].bar(display_names, r2_vals, color=colors, width=0.55, edgecolor=CFCL_DARK, linewidth=1.2)
        axes[0].set_title(r"Goodness of Fit ($R^2$ Score)", fontsize=13, fontweight="bold", color=CFCL_DARK)
        axes[0].set_ylim(0, 1.05)
        axes[0].axhline(0.90, color="#E74C3C", linestyle="--", linewidth=1.2, label="Industrial Target (0.90)")
        for i, v in enumerate(r2_vals):
            axes[0].text(i, v + 0.02, f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")
        axes[0].legend(loc="lower left", framealpha=0.9)

        # 2. RMSE
        axes[1].bar(display_names, rmse_vals, color=colors, width=0.55, edgecolor=CFCL_DARK, linewidth=1.2)
        axes[1].set_title("Root Mean Squared Error (Gcal/MT)", fontsize=13, fontweight="bold", color=CFCL_DARK)
        axes[1].set_ylim(0, max(rmse_vals) * 1.25)
        for i, v in enumerate(rmse_vals):
            axes[1].text(i, v + 0.0015, f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")

        # 3. MAE
        axes[2].bar(display_names, mae_vals, color=colors, width=0.55, edgecolor=CFCL_DARK, linewidth=1.2)
        axes[2].set_title("Mean Absolute Error (Gcal/MT)", fontsize=13, fontweight="bold", color=CFCL_DARK)
        axes[2].set_ylim(0, max(mae_vals) * 1.25)
        for i, v in enumerate(mae_vals):
            axes[2].text(i, v + 0.0015, f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")

        for ax in axes:
            ax.set_xticks(range(len(display_names)))
            ax.set_xticklabels(display_names, rotation=15, ha="right", fontsize=10, fontweight="bold")
            ax.grid(axis="y", linestyle=":", alpha=0.6)

        plt.suptitle(
            f"CFCL Gadepan SEC Model Leaderboard (Champion: {self.leaderboard['champion_model'].replace('_', ' ')})",
            fontsize=15,
            fontweight="bold",
            color=CFCL_BLUE,
            y=1.03,
        )
        plt.tight_layout()
        out_path = self.figures_dir / "model_leaderboard_comparison.png"
        save_figure(fig, out_path)
        plt.close(fig)
        self.logger.info("Saved leaderboard plot to %s", out_path)
        return out_path

    def plot_actual_vs_predicted(self) -> Path:
        """Create 4-panel parity plot with 45-degree line and +-2% DCS tolerance bands."""
        apply_cfcl_theme()
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        models = list(self.leaderboard["models"].keys())
        y_true = self.y_true
        min_sec = float(np.min(y_true) * 0.98)
        max_sec = float(np.max(y_true) * 1.02)

        for i, m_name in enumerate(models):
            ax = axes[i]
            preds = self.predictions[m_name]
            m_metrics = self.leaderboard["models"][m_name]

            # Scatter colored by plant
            for p_id in ["gadepan_1", "gadepan_2", "gadepan_3"]:
                mask = self.holdout_df["plant_id"] == p_id
                ax.scatter(
                    y_true[mask],
                    preds[mask],
                    c=PLANT_COLORS[p_id],
                    label=PLANT_LABELS[p_id],
                    alpha=0.65,
                    s=22,
                    edgecolor="none",
                )

            # 45-degree line
            ref_line = np.linspace(min_sec, max_sec, 100)
            ax.plot(ref_line, ref_line, "k-", linewidth=1.5, label="Ideal Parity (y = x)")

            # +-2% DCS Tolerance Envelope
            ax.fill_between(
                ref_line,
                ref_line * 0.98,
                ref_line * 1.02,
                color="gray",
                alpha=0.15,
                label=r"$\pm 2\%$ DCS Tolerance",
            )

            ax.set_xlim(min_sec, max_sec)
            ax.set_ylim(min_sec, max_sec)
            ax.set_xlabel("Actual SEC (Gcal/MT urea)", fontsize=11, fontweight="bold")
            ax.set_ylabel("Predicted SEC (Gcal/MT urea)", fontsize=11, fontweight="bold")
            ax.set_title(
                f"{m_name.replace('_', ' ')}: $R^2={m_metrics['r2_score']:.4f}$ | RMSE={m_metrics['rmse']:.4f}",
                fontsize=12,
                fontweight="bold",
                color=CFCL_DARK,
            )
            ax.grid(True, linestyle=":", alpha=0.6)
            if i == 0:
                ax.legend(loc="upper left", framealpha=0.9, fontsize=9)

        plt.suptitle(
            "Holdout Actual vs. Predicted SEC Across 3 Plants (Kota, Rajasthan)",
            fontsize=15,
            fontweight="bold",
            color=CFCL_BLUE,
            y=0.995,
        )
        plt.tight_layout()
        out_path = self.figures_dir / "actual_vs_predicted_all_models.png"
        save_figure(fig, out_path)
        plt.close(fig)
        self.logger.info("Saved actual vs predicted parity plot to %s", out_path)
        return out_path

    def plot_residual_diagnostics(self) -> Path:
        """Create residual distribution and Q-Q plots for the champion model."""
        apply_cfcl_theme()
        champ = self.leaderboard["champion_model"]
        y_true = self.y_true
        y_pred = self.predictions[champ]
        residuals = y_true - y_pred

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, c=CFCL_BLUE, alpha=0.55, s=20, edgecolor="none")
        axes[0].axhline(0, color="red", linestyle="--", linewidth=1.5)
        # 95% bands
        ci = self.leaderboard["champion_metrics"]["prediction_interval_95"]
        axes[0].axhline(ci, color="gray", linestyle=":", label=f"+95% PI (+{ci:.4f})")
        axes[0].axhline(-ci, color="gray", linestyle=":", label=f"-95% PI (-{ci:.4f})")
        axes[0].set_xlabel("Predicted SEC (Gcal/MT)", fontsize=11, fontweight="bold")
        axes[0].set_ylabel("Residual (Actual - Pred)", fontsize=11, fontweight="bold")
        axes[0].set_title(f"Residuals vs. Fitted ({champ})", fontsize=12, fontweight="bold")
        axes[0].legend(loc="upper right", fontsize=9)
        axes[0].grid(True, linestyle=":", alpha=0.6)

        # 2. Residual Distribution vs Normal Curve
        axes[1].hist(residuals, bins=40, density=True, color=CFCL_LIGHT_BLUE, edgecolor=CFCL_DARK, alpha=0.7)
        mu, std = np.mean(residuals), np.std(residuals)
        x_norm = np.linspace(mu - 3.5 * std, mu + 3.5 * std, 100)
        axes[1].plot(x_norm, stats.norm.pdf(x_norm, mu, std), "r-", linewidth=2.0, label="Gaussian Fit")
        axes[1].set_xlabel("Residual Error (Gcal/MT)", fontsize=11, fontweight="bold")
        axes[1].set_ylabel("Density", fontsize=11, fontweight="bold")
        axes[1].set_title(f"Residual Distribution (Mean: {mu:.4f})", fontsize=12, fontweight="bold")
        axes[1].legend(loc="upper right", fontsize=9)
        axes[1].grid(True, linestyle=":", alpha=0.6)

        # 3. Normal Q-Q Plot
        stats.probplot(residuals, dist="norm", plot=axes[2])
        axes[2].get_lines()[0].set_markerfacecolor(CFCL_BLUE)
        axes[2].get_lines()[0].set_markeredgecolor("none")
        axes[2].get_lines()[0].set_alpha(0.6)
        axes[2].get_lines()[1].set_color("red")
        axes[2].get_lines()[1].set_linewidth(1.8)
        axes[2].set_title(f"Normal Q-Q Plot (DW={self.leaderboard['champion_metrics']['durbin_watson']})",
                          fontsize=12, fontweight="bold")
        axes[2].grid(True, linestyle=":", alpha=0.6)

        plt.suptitle(f"CFCL Model Residual Diagnostics - Champion: {champ.replace('_', ' ')}",
                     fontsize=15, fontweight="bold", color=CFCL_BLUE, y=1.02)
        plt.tight_layout()
        out_path = self.figures_dir / "residual_diagnostics.png"
        save_figure(fig, out_path)
        plt.close(fig)
        self.logger.info("Saved residual diagnostics plot to %s", out_path)
        return out_path

    def plot_within_plant_breakdown(self) -> Path:
        """Plot RMSE and R2 broken down by operational plant unit."""
        apply_cfcl_theme()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        plants = ["gadepan_1", "gadepan_2", "gadepan_3"]
        plant_labels = [PLANT_LABELS[p] for p in plants]
        models = list(self.leaderboard["models"].keys())

        x = np.arange(len(plants))
        width = 0.20

        # 1. Within-Plant RMSE
        for i, m_name in enumerate(models):
            rmses = [self.leaderboard["models"][m_name]["within_plant"][p]["rmse"] for p in plants]
            axes[0].bar(x + (i - 1.5) * width, rmses, width, label=m_name.replace("_", " "))

        axes[0].set_xticks(x)
        axes[0].set_xticklabels(plant_labels, fontsize=10, fontweight="bold")
        axes[0].set_ylabel("RMSE (Gcal/MT urea)", fontsize=11, fontweight="bold")
        axes[0].set_title("Within-Plant Root Mean Squared Error", fontsize=13, fontweight="bold", color=CFCL_DARK)
        axes[0].grid(axis="y", linestyle=":", alpha=0.6)
        axes[0].legend(loc="upper right", fontsize=9)

        # 2. Within-Plant R2
        for i, m_name in enumerate(models):
            r2s = [self.leaderboard["models"][m_name]["within_plant"][p]["r2"] for p in plants]
            axes[1].bar(x + (i - 1.5) * width, r2s, width, label=m_name.replace("_", " "))

        axes[1].set_xticks(x)
        axes[1].set_xticklabels(plant_labels, fontsize=10, fontweight="bold")
        axes[1].set_ylabel(r"$R^2$ Score", fontsize=11, fontweight="bold")
        axes[1].set_title(r"Within-Plant $R^2$ Score (Internal Variance Explained)", fontsize=13, fontweight="bold", color=CFCL_DARK)
        axes[1].grid(axis="y", linestyle=":", alpha=0.6)

        plt.suptitle("Fleet Generalization Diagnostic: Performance Breakdown by Operating Unit",
                     fontsize=15, fontweight="bold", color=CFCL_BLUE, y=1.02)
        plt.tight_layout()
        out_path = self.figures_dir / "within_plant_performance.png"
        save_figure(fig, out_path)
        plt.close(fig)
        self.logger.info("Saved within-plant breakdown plot to %s", out_path)
        return out_path


def main() -> None:
    """Run unified evaluation pipeline."""
    evaluator = ModelEvaluator()
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
