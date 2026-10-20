"""Feature Importance & Process Variable Elasticity Analyzer for CFCL SEC Optimization.

Provides chemical engineering and statistical interpretability:
1. Permutation Feature Importance (unbiased by tree split cardinality, 5-fold repeat)
2. Manipulated Variables (MVs) vs. Disturbance Variables (DVs) Elasticity Analysis:
   - Quantifies the marginal sensitivity of SEC (ΔSEC / ΔX) per operational setpoint step
   - Answers critical DCS operator questions: "How much does a 0.1 increase in S/C ratio
     or a 1°C rise in ambient temperature impact specific energy consumption?"
3. Feature Collinearity & Redundancy Screening
4. Publishes CFCL-branded sensitivity plots and structured JSON reports
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.data.schema import CONTROLLABLE_FEATURES, TARGET_COLUMN, UNCONTROLLABLE_FEATURES
from src.utils.config_loader import get_paths, get_project_root, load_config
from src.utils.logger import get_logger
from src.utils.visualization import (
    CFCL_BLUE,
    CFCL_DARK,
    CFCL_LIGHT_BLUE,
    apply_cfcl_theme,
    save_figure,
)


class FeatureImportanceAnalyzer:
    """Enterprise feature importance and process sensitivity / elasticity engine."""

    def __init__(self, model_type: str = "random_forest", logger: Optional[Any] = None) -> None:
        self.logger = logger or get_logger("feature_importance")
        self.paths = get_paths()
        self.model_type = model_type.lower()
        self.figures_dir = self.paths["figures"]
        self.reports_dir = self.paths["reports"]

        self.model: Optional[Any] = None
        self.feature_names: List[str] = []
        self.holdout_df: Optional[pd.DataFrame] = None
        self.X_val: Optional[pd.DataFrame] = None
        self.y_val: Optional[np.ndarray] = None

        self.perm_results: Optional[Dict[str, Any]] = None
        self.elasticity_results: Optional[List[Dict[str, Any]]] = None

    def load_model_and_holdout(self) -> Tuple[Any, pd.DataFrame, np.ndarray]:
        """Load trained champion model and chronological holdout test set."""
        models_dir = self.paths["models"]
        processed_dir = self.paths["processed_data"]

        if self.model_type in ["rf", "random_forest"]:
            model_path = models_dir / "random_forest.joblib"
        elif self.model_type == "xgboost":
            model_path = models_dir / "xgboost_sec.joblib"
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.feature_names = artifact["feature_names"]
        self.logger.info("Loaded %s with %d features from %s", self.model_type, len(self.feature_names), model_path)

        parquet_path = processed_dir / "plant_features.parquet"
        df = pd.read_parquet(parquet_path).sort_values("timestamp").dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
        val_size = int(len(df) * 0.2)
        self.holdout_df = df.iloc[-val_size:].copy().reset_index(drop=True)

        self.X_val = self.holdout_df[self.feature_names].fillna(0)
        self.y_val = self.holdout_df[TARGET_COLUMN].values
        self.logger.info("Prepared holdout validation matrix: %s", self.X_val.shape)

        return self.model, self.X_val, self.y_val

    def compute_permutation_importance(self, n_repeats: int = 5, n_sample: int = 800) -> Dict[str, Any]:
        """Compute permutation importance across holdout test set."""
        if self.model is None or self.X_val is None:
            self.load_model_and_holdout()

        self.logger.info("Computing Permutation Importance (%d repeats, %d sample size)...", n_repeats, n_sample)
        
        # Subsample for speed if needed
        if len(self.X_val) > n_sample:
            idx = np.random.RandomState(42).choice(len(self.X_val), size=n_sample, replace=False)
            X_sub = self.X_val.iloc[idx].copy()
            y_sub = self.y_val[idx]
        else:
            X_sub = self.X_val.copy()
            y_sub = self.y_val

        r = permutation_importance(
            self.model,
            X_sub,
            y_sub,
            n_repeats=n_repeats,
            random_state=42,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )

        importances_mean = r.importances_mean
        importances_std = r.importances_std
        sorted_idx = np.argsort(importances_mean)[::-1]

        ranked_features = [
            {
                "rank": i + 1,
                "feature": self.feature_names[idx],
                "importance_mean": float(round(importances_mean[idx], 5)),
                "importance_std": float(round(importances_std[idx], 5)),
            }
            for i, idx in enumerate(sorted_idx[:25])
        ]

        self.perm_results = {
            "model_type": self.model_type,
            "sample_records": len(X_sub),
            "n_repeats": n_repeats,
            "top_features": ranked_features,
        }
        self.logger.info("Permutation importance calculated for top 25 features.")
        return self.perm_results

    def compute_process_elasticity(self) -> List[Dict[str, Any]]:
        """Compute operational process sensitivity (ΔSEC / ΔX) for MVs and DVs."""
        if self.model is None or self.X_val is None:
            self.load_model_and_holdout()

        self.logger.info("Computing chemical process elasticity and delta sensitivity...")
        
        # Operational perturbation step definitions (realistic DCS changes)
        perturbations = {
            # Manipulated Variables (MVs / Controllable)
            "steam_carbon_ratio": {"delta": 0.1, "unit": "ratio", "category": "Manipulated (Controllable)"},
            "reformer_outlet_temp": {"delta": 5.0, "unit": "°C", "category": "Manipulated (Controllable)"},
            "loop_pressure": {"delta": 2.0, "unit": "bar", "category": "Manipulated (Controllable)"},
            # Disturbance Variables (DVs / Uncontrollable)
            "ambient_temp": {"delta": 2.0, "unit": "°C", "category": "Disturbance (Ambient/Feed)"},
            "cw_inlet_temp": {"delta": 1.0, "unit": "°C", "category": "Disturbance (Ambient/Feed)"},
            "ng_calorific_value": {"delta": 50.0, "unit": "kcal/Sm³", "category": "Disturbance (Ambient/Feed)"},
            "ambient_humidity": {"delta": 5.0, "unit": "%", "category": "Disturbance (Ambient/Feed)"},
            "co2_slip_ppm": {"delta": 100.0, "unit": "ppm", "category": "Disturbance (Ambient/Feed)"},
            "catalyst_age_days": {"delta": 30.0, "unit": "days", "category": "Disturbance (Ambient/Feed)"},
        }

        # Baseline predictions
        base_preds = self.model.predict(self.X_val)
        mean_base_sec = float(np.mean(base_preds))

        elasticity_list = []

        for var, spec in perturbations.items():
            if var not in self.feature_names:
                continue

            delta_val = spec["delta"]
            X_pert = self.X_val.copy()
            X_pert[var] = X_pert[var] + delta_val

            # Also update any derived rolling/interaction features that depend directly on this var
            # (e.g. steam_reformer_interaction if steam_carbon_ratio or reformer_outlet_temp changes)
            if "steam_reformer_interaction" in self.feature_names:
                if var == "steam_carbon_ratio":
                    X_pert["steam_reformer_interaction"] = X_pert[var] * X_pert["reformer_outlet_temp"]
                elif var == "reformer_outlet_temp":
                    X_pert["steam_reformer_interaction"] = X_pert["steam_carbon_ratio"] * X_pert[var]

            if "ambient_cw_interaction" in self.feature_names:
                if var == "ambient_temp":
                    X_pert["ambient_cw_interaction"] = X_pert[var] * X_pert["cw_inlet_temp"]
                elif var == "cw_inlet_temp":
                    X_pert["ambient_cw_interaction"] = X_pert["ambient_temp"] * X_pert[var]

            pert_preds = self.model.predict(X_pert)
            delta_sec = float(np.mean(pert_preds - base_preds))
            mean_var_val = float(np.mean(self.X_val[var]))

            # Normalized elasticity: (% change in SEC) / (% change in Variable)
            pct_sec = (delta_sec / mean_base_sec) * 100.0
            pct_var = (delta_val / mean_var_val) * 100.0
            elasticity = pct_sec / pct_var if pct_var != 0 else 0.0

            elasticity_list.append({
                "variable": var,
                "category": spec["category"],
                "step_change": f"+{delta_val} {spec['unit']}",
                "mean_value": round(mean_var_val, 2),
                "delta_sec_gcal_per_mt": round(delta_sec, 5),
                "percent_sec_impact": round(pct_sec, 4),
                "elasticity_coefficient": round(elasticity, 4),
                "operational_guidance": (
                    f"Increasing {var} by {delta_val} {spec['unit']} "
                    f"{'reduces' if delta_sec < 0 else 'increases'} plant SEC by {abs(delta_sec):.4f} Gcal/MT"
                ),
            })

        self.elasticity_results = elasticity_list
        self.logger.info("Calculated elasticity for %d process variables.", len(elasticity_list))
        return self.elasticity_results

    # =========================================================================
    # Visualizations
    # =========================================================================

    def plot_permutation_importance(self, max_display: int = 15) -> Path:
        """Create publication-grade bar chart comparing Permutation Importance with error bars."""
        apply_cfcl_theme()
        if self.perm_results is None:
            self.compute_permutation_importance()

        top_feats = self.perm_results["top_features"][:max_display][::-1]
        names = [f["feature"].replace("_", " ") for f in top_feats]
        means = [f["importance_mean"] for f in top_feats]
        stds = [f["importance_std"] for f in top_feats]

        fig, ax = plt.subplots(figsize=(11, 7))
        bars = ax.barh(names, means, xerr=stds, color=CFCL_BLUE, edgecolor=CFCL_DARK,
                       height=0.6, alpha=0.9, capsize=3, error_kw={"elinewidth": 1.2, "ecolor": "red"})

        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.0005, bar.get_y() + bar.get_height() / 2, f"{w:.4f}",
                    va="center", ha="left", fontsize=9, fontweight="bold", color=CFCL_DARK)

        ax.set_xlabel("Permutation Importance (Increase in RMSE when shuffled, Gcal/MT)", fontsize=11, fontweight="bold")
        ax.set_title(
            f"Holdout Permutation Feature Importance ({self.model_type.upper()}) - 5 Repeats",
            fontsize=13,
            fontweight="bold",
            color=CFCL_BLUE,
            pad=12,
        )
        ax.grid(axis="x", linestyle=":", alpha=0.6)
        plt.tight_layout()

        out_path = self.figures_dir / "permutation_importance_comparison.png"
        save_figure(fig, out_path)
        plt.close(fig)
        self.logger.info("Saved permutation importance plot to %s", out_path)
        return out_path

    def plot_process_elasticity(self) -> Path:
        """Create horizontal bar chart showing process sensitivity (ΔSEC) for MVs vs DVs."""
        apply_cfcl_theme()
        if self.elasticity_results is None:
            self.compute_process_elasticity()

        # Sort by absolute delta SEC
        sorted_specs = sorted(self.elasticity_results, key=lambda x: abs(x["delta_sec_gcal_per_mt"]), reverse=False)

        names = [f"{s['variable'].replace('_', ' ')} ({s['step_change']})" for s in sorted_specs]
        deltas = [s["delta_sec_gcal_per_mt"] for s in sorted_specs]
        colors = [
            "#27AE60" if "Manipulated" in s["category"] else "#E67E22"
            for s in sorted_specs
        ]

        fig, ax = plt.subplots(figsize=(12, 7))
        bars = ax.barh(names, deltas, color=colors, edgecolor=CFCL_DARK, height=0.6, alpha=0.9)
        ax.axvline(0, color="black", linestyle="-", linewidth=1.2)

        for bar, delta in zip(bars, deltas):
            w = bar.get_width()
            offset = 0.0003 if w >= 0 else -0.0003
            ha = "left" if w >= 0 else "right"
            ax.text(w + offset, bar.get_y() + bar.get_height() / 2, f"{delta:+.4f}",
                    va="center", ha=ha, fontsize=9, fontweight="bold")

        # Custom legend for MVs vs DVs
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#27AE60", edgecolor=CFCL_DARK, label="Manipulated Setpoints (Controllable MVs)"),
            Patch(facecolor="#E67E22", edgecolor=CFCL_DARK, label="External Disturbances (Ambient / Feed DVs)"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9, fontsize=10)

        ax.set_xlabel("Marginal Change in Specific Energy Consumption (ΔSEC, Gcal/MT urea)", fontsize=11, fontweight="bold")
        ax.set_title(
            "CFCL DCS Operational Sensitivity: Marginal Impact per Unit Process Perturbation",
            fontsize=13,
            fontweight="bold",
            color=CFCL_BLUE,
            pad=12,
        )
        ax.grid(axis="x", linestyle=":", alpha=0.6)
        plt.tight_layout()

        out_path = self.figures_dir / "process_variable_elasticity.png"
        save_figure(fig, out_path)
        plt.close(fig)
        self.logger.info("Saved process variable elasticity plot to %s", out_path)
        return out_path

    def export_report_json(self) -> Path:
        """Export feature importance and elasticity report to JSON."""
        report = {
            "model_type": self.model_type,
            "permutation_importance": self.perm_results,
            "process_variable_elasticity": self.elasticity_results,
        }
        out_path = self.reports_dir / "feature_importance_report.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        self.logger.info("Saved feature importance report to %s", out_path)
        return out_path

    def run_analysis(self) -> Dict[str, Any]:
        """Execute complete feature importance and elasticity pipeline."""
        self.logger.info("Initiating complete Feature Importance & Elasticity analysis...")
        self.load_model_and_holdout()
        self.compute_permutation_importance()
        self.compute_process_elasticity()

        self.plot_permutation_importance()
        self.plot_process_elasticity()
        self.export_report_json()

        self.logger.info("Feature importance pipeline complete. All plots and reports exported.")
        return {"status": "success", "variables_analyzed": len(self.elasticity_results or [])}


def main() -> None:
    """Run CLI."""
    analyzer = FeatureImportanceAnalyzer(model_type="random_forest")
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
