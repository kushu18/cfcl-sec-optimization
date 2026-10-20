"""SHAP (SHapley Additive exPlanations) Analyzer for CFCL SEC Optimization.

Provides rigorous, game-theoretic interpretability for tree-based process models
(XGBoost and Random Forest):
- Global feature attribution via Beeswarm and Mean Absolute SHAP ranking
- Local excursion decomposition via Waterfall plots (identifying root causes of high-SEC spikes)
- Thermodynamic 2D interaction dependence surfaces (Steam-to-Carbon vs Reformer Temp,
  Ambient Temp vs Cooling Water Inlet Temp)
- Safe serialization of SHAP metrics and attributions for DCS operators and engineers
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import shap.explainers._tree as tree_mod

from src.data.schema import TARGET_COLUMN
from src.utils.config_loader import get_paths, get_project_root, load_config
from src.utils.logger import get_logger
from src.utils.visualization import (
    CFCL_BLUE,
    CFCL_DARK,
    CFCL_LIGHT_BLUE,
    apply_cfcl_theme,
    save_figure,
)


def _patch_shap_xgboost_loader() -> None:
    """Patch SHAP 0.45.1 XGBTreeModelLoader to support XGBoost 3.x base_score format."""
    orig_init = tree_mod.XGBTreeModelLoader.__init__

    def patched_init(self: Any, xgb_model: Any) -> None:
        raw = xgb_model.save_raw(raw_format="ubj")
        with io.BytesIO(raw) as fd:
            jmodel = tree_mod.decode_ubjson_buffer(fd)
        
        learner = jmodel.get("learner", {})
        param = learner.get("learner_model_param", {})
        bs = param.get("base_score")
        if isinstance(bs, str):
            param["base_score"] = re.sub(r"[\[\]]", "", bs)

        orig_decode = tree_mod.decode_ubjson_buffer
        tree_mod.decode_ubjson_buffer = lambda _: jmodel
        try:
            orig_init(self, xgb_model)
        finally:
            tree_mod.decode_ubjson_buffer = orig_decode

    tree_mod.XGBTreeModelLoader.__init__ = patched_init


# Apply patch at module import
_patch_shap_xgboost_loader()


class SHAPAnalyzer:
    """Industrial SHAP explainability engine for tree-based SEC models."""

    def __init__(self, model_type: str = "xgboost", logger: Optional[Any] = None) -> None:
        self.logger = logger or get_logger("shap_analyzer")
        self.paths = get_paths()
        self.model_cfg = load_config("model")
        self.shap_cfg = self.model_cfg.get("shap", {})
        self.model_type = model_type.lower()

        self.model: Optional[Any] = None
        self.feature_names: List[str] = []
        self.explainer: Optional[shap.TreeExplainer] = None
        self.shap_values: Optional[shap.Explanation] = None
        self.X_sample: Optional[pd.DataFrame] = None
        self.df_sample: Optional[pd.DataFrame] = None

    def load_model_and_data(self, sample_size: int = 600) -> Tuple[Any, pd.DataFrame]:
        """Load trained tree model and representative holdout sample for attribution."""
        models_dir = self.paths["models"]
        processed_dir = self.paths["processed_data"]

        if self.model_type == "xgboost":
            model_path = models_dir / "xgboost_sec.joblib"
        elif self.model_type in ["rf", "random_forest"]:
            model_path = models_dir / "random_forest.joblib"
        else:
            raise ValueError(f"Unsupported model type for TreeExplainer: {self.model_type}")

        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {model_path}")

        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.feature_names = artifact["feature_names"]
        self.logger.info("Loaded %s model with %d features from %s", self.model_type, len(self.feature_names), model_path)

        # Load holdout dataset
        parquet_path = processed_dir / "plant_features.parquet"
        df = pd.read_parquet(parquet_path).sort_values("timestamp").dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
        
        # Take holdout portion (last 20%)
        val_size = int(len(df) * 0.2)
        holdout = df.iloc[-val_size:].copy().reset_index(drop=True)

        # Sample representative rows across plants
        if len(holdout) > sample_size:
            sample_df = holdout.sample(n=sample_size, random_state=42).sort_values("timestamp").reset_index(drop=True)
        else:
            sample_df = holdout.copy()

        self.df_sample = sample_df
        self.X_sample = sample_df[self.feature_names].fillna(0)
        self.logger.info("Prepared SHAP background evaluation set: %d records", len(self.X_sample))

        return self.model, self.X_sample

    def compute_shap_values(self) -> shap.Explanation:
        """Compute exact TreeExplainer SHAP values matrix."""
        if self.model is None or self.X_sample is None:
            self.load_model_and_data()

        self.logger.info("Initializing shap.TreeExplainer on %s...", self.model_type)
        self.explainer = shap.TreeExplainer(self.model)
        
        self.logger.info("Calculating SHAP values across %d features...", self.X_sample.shape[1])
        self.shap_values = self.explainer(self.X_sample)
        self.logger.info("Computed SHAP matrix: shape=%s, base_value=%.4f Gcal/MT",
                         self.shap_values.shape, float(np.mean(self.shap_values.base_values)))
        return self.shap_values

    def get_top_features(self, n_features: int = 15) -> List[Tuple[str, float]]:
        """Return top N features ranked by mean absolute SHAP value."""
        if self.shap_values is None:
            self.compute_shap_values()

        mean_abs_shap = np.mean(np.abs(self.shap_values.values), axis=0)
        ranked_indices = np.argsort(mean_abs_shap)[::-1][:n_features]
        return [(self.feature_names[i], float(mean_abs_shap[i])) for i in ranked_indices]

    # =========================================================================
    # Visualizations
    # =========================================================================

    def plot_summary_beeswarm(self, max_display: int = 15) -> Path:
        """Create publication-quality SHAP beeswarm summary plot."""
        apply_cfcl_theme()
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Standard SHAP beeswarm
        shap.summary_plot(
            self.shap_values.values,
            self.X_sample,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
            plot_type="dot",
        )
        
        plt.title(
            f"CFCL Gadepan - SHAP Feature Attribution Impact on SEC ({self.model_type.upper()})",
            fontsize=13,
            fontweight="bold",
            color=CFCL_BLUE,
            pad=15,
        )
        plt.xlabel("SHAP Value (Impact on Specific Energy Consumption, Gcal/MT urea)", fontsize=11, fontweight="bold")
        plt.tight_layout()

        out_path = self.paths["figures"] / "shap_summary_beeswarm.png"
        save_figure(plt.gcf(), out_path)
        plt.close("all")
        self.logger.info("Saved SHAP beeswarm plot to %s", out_path)
        return out_path

    def plot_feature_importance_bar(self, max_display: int = 15) -> Path:
        """Create clean CFCL-branded horizontal bar chart of mean absolute SHAP values."""
        apply_cfcl_theme()
        top_feats = self.get_top_features(max_display)
        names = [f[0].replace("_", " ") for f in top_feats][::-1]
        vals = [f[1] for f in top_feats][::-1]

        fig, ax = plt.subplots(figsize=(11, 7))
        bars = ax.barh(names, vals, color=CFCL_BLUE, edgecolor=CFCL_DARK, height=0.65, alpha=0.9)
        
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.001, bar.get_y() + bar.get_height() / 2, f"{w:.4f}",
                    va="center", ha="left", fontsize=9, fontweight="bold", color=CFCL_DARK)

        ax.set_xlabel("Mean Absolute SHAP Value (|Gcal/MT urea|)", fontsize=11, fontweight="bold")
        ax.set_title(
            f"Top {max_display} Drivers of Specific Energy Consumption (Game-Theoretic SHAP)",
            fontsize=13,
            fontweight="bold",
            color=CFCL_BLUE,
            pad=12,
        )
        ax.grid(axis="x", linestyle=":", alpha=0.6)
        plt.tight_layout()

        out_path = self.paths["figures"] / "shap_feature_importance_bar.png"
        save_figure(fig, out_path)
        plt.close(fig)
        self.logger.info("Saved SHAP importance bar chart to %s", out_path)
        return out_path

    def plot_waterfall_excursion(self) -> Tuple[Path, Path]:
        """Plot local waterfall decompositions for the worst operational excursion and best benchmark hour."""
        apply_cfcl_theme()
        y_true = self.df_sample[TARGET_COLUMN].values
        
        # Worst hour (highest SEC excursion spike)
        worst_idx = int(np.argmax(y_true))
        worst_row = self.df_sample.iloc[worst_idx]
        worst_sec = y_true[worst_idx]

        fig, ax = plt.subplots(figsize=(12, 7))
        shap.plots.waterfall(self.shap_values[worst_idx], max_display=12, show=False)
        plt.title(
            f"Excursion Root-Cause Decomposition: {worst_row['plant_id']} at {worst_row['timestamp']}\n"
            f"Actual SEC: {worst_sec:.3f} Gcal/MT (High Energy Excursion Event)",
            fontsize=12,
            fontweight="bold",
            color="#C0392B",
            pad=12,
        )
        plt.tight_layout()
        worst_path = self.paths["figures"] / "shap_waterfall_worst_hour.png"
        save_figure(plt.gcf(), worst_path)
        plt.close("all")

        # Best hour (lowest SEC benchmark operation)
        best_idx = int(np.argmin(y_true))
        best_row = self.df_sample.iloc[best_idx]
        best_sec = y_true[best_idx]

        fig, ax = plt.subplots(figsize=(12, 7))
        shap.plots.waterfall(self.shap_values[best_idx], max_display=12, show=False)
        plt.title(
            f"Energy Benchmark Decomposition: {best_row['plant_id']} at {best_row['timestamp']}\n"
            f"Actual SEC: {best_sec:.3f} Gcal/MT (Optimal High-Efficiency Operation)",
            fontsize=12,
            fontweight="bold",
            color="#27AE60",
            pad=12,
        )
        plt.tight_layout()
        best_path = self.paths["figures"] / "shap_waterfall_best_hour.png"
        save_figure(plt.gcf(), best_path)
        plt.close("all")

        self.logger.info("Saved local waterfall plots to %s and %s", worst_path, best_path)
        return worst_path, best_path

    def plot_thermodynamic_dependence(self) -> Tuple[Path, Path]:
        """Plot 2D SHAP dependence plots for critical chemical process interactions."""
        apply_cfcl_theme()

        # 1. Steam-to-Carbon Ratio vs Reformer Temperature
        fig, ax = plt.subplots(figsize=(10, 6))
        sc_col = "steam_carbon_ratio" if "steam_carbon_ratio" in self.feature_names else self.feature_names[0]
        ref_col = "reformer_outlet_temp" if "reformer_outlet_temp" in self.feature_names else None

        shap.dependence_plot(
            sc_col,
            self.shap_values.values,
            self.X_sample,
            feature_names=self.feature_names,
            interaction_index=ref_col,
            show=False,
            ax=ax,
        )
        ax.set_title(
            "Thermodynamic Coupling: Steam-to-Carbon Ratio vs. Reformer Outlet Temp",
            fontsize=12,
            fontweight="bold",
            color=CFCL_BLUE,
            pad=10,
        )
        ax.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        dep1_path = self.paths["figures"] / "shap_dependence_steam_reformer.png"
        save_figure(fig, dep1_path)
        plt.close(fig)

        # 2. Ambient Temperature vs Cooling Water Inlet Temperature
        fig, ax = plt.subplots(figsize=(10, 6))
        amb_col = "ambient_temp" if "ambient_temp" in self.feature_names else self.feature_names[1]
        cw_col = "cw_inlet_temp" if "cw_inlet_temp" in self.feature_names else None

        shap.dependence_plot(
            amb_col,
            self.shap_values.values,
            self.X_sample,
            feature_names=self.feature_names,
            interaction_index=cw_col,
            show=False,
            ax=ax,
        )
        ax.set_title(
            "Rajasthan Summer Disturbance: Ambient Temperature vs. Cooling Water Inlet Temp",
            fontsize=12,
            fontweight="bold",
            color=CFCL_BLUE,
            pad=10,
        )
        ax.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        dep2_path = self.paths["figures"] / "shap_dependence_ambient_cw.png"
        save_figure(fig, dep2_path)
        plt.close(fig)

        self.logger.info("Saved thermodynamic dependence plots to %s and %s", dep1_path, dep2_path)
        return dep1_path, dep2_path

    def export_summary_json(self) -> Path:
        """Export structured SHAP attributions and engineering summary to reports."""
        top_15 = self.get_top_features(15)
        
        # Determine directional impact (+ or - correlation with SEC)
        correlations: Dict[str, float] = {}
        for feat, _ in top_15:
            feat_idx = self.feature_names.index(feat)
            shap_f = self.shap_values.values[:, feat_idx]
            val_f = self.X_sample[feat].values
            if np.std(shap_f) > 0 and np.std(val_f) > 0:
                corr = float(np.corrcoef(shap_f, val_f)[0, 1])
            else:
                corr = 0.0
            correlations[feat] = round(corr, 4)

        report = {
            "model_analyzed": self.model_type,
            "sample_records": len(self.X_sample),
            "base_value_sec_gcal_per_mt": round(float(np.mean(self.shap_values.base_values)), 4),
            "top_features": [
                {
                    "rank": i + 1,
                    "feature": feat,
                    "mean_abs_shap": round(score, 5),
                    "directional_correlation": correlations.get(feat, 0.0),
                    "interpretation": (
                        "Higher values increase energy consumption (undesirable)"
                        if correlations.get(feat, 0.0) > 0.1
                        else "Higher values reduce energy consumption (favorable)"
                        if correlations.get(feat, 0.0) < -0.1
                        else "Complex non-linear sweet-spot behavior"
                    ),
                }
                for i, (feat, score) in enumerate(top_15)
            ],
            "worst_hour_excursion": {
                "timestamp": str(self.df_sample.loc[int(np.argmax(self.df_sample[TARGET_COLUMN].values)), "timestamp"]),
                "plant_id": str(self.df_sample.loc[int(np.argmax(self.df_sample[TARGET_COLUMN].values)), "plant_id"]),
                "peak_sec": float(np.max(self.df_sample[TARGET_COLUMN].values)),
            },
        }

        out_path = self.paths["reports"] / "shap_summary.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        self.logger.info("Saved SHAP summary report to %s", out_path)
        return out_path

    def run_analysis(self) -> Dict[str, Any]:
        """Execute full SHAP explainability pipeline and generate all figures."""
        self.logger.info("Starting complete SHAP analysis pipeline...")
        self.load_model_and_data()
        self.compute_shap_values()
        
        self.plot_summary_beeswarm()
        self.plot_feature_importance_bar()
        self.plot_waterfall_excursion()
        self.plot_thermodynamic_dependence()
        self.export_summary_json()
        
        self.logger.info("SHAP analysis pipeline complete. All artifacts generated successfully.")
        return {"status": "success", "model": self.model_type, "top_features": self.get_top_features(5)}


def main() -> None:
    """Run SHAP analyzer CLI."""
    analyzer = SHAPAnalyzer(model_type="xgboost")
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
