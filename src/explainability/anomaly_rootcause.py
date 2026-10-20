"""DCS Operational Excursion Detection & SHAP Root-Cause Decomposer.

Provides real-time and post-mortem incident diagnostics:
1. Multi-tier statistical anomaly detection (2σ mild excursion, 3σ severe trip).
2. Local SHAP game-theoretic decomposition attributing energy surges to specific
   controllable setpoints, feed gas disruptions, or ambient heatwaves.
3. Automated incident failure-mode classification:
   - Heat Stress & Condenser Vacuum Degradation
   - Reforming Stoichiometry & Catalyst Coking Risk
   - Synthesis Loop Compressor Surge
   - CO₂ Absorption Breakthrough
   - Natural Gas Pipeline Quality Shock
4. Publication-quality timeline scatter plots and waterfall incident breakdowns.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.data.schema import CONTROLLABLE_FEATURES, TARGET_COLUMN
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


@dataclass
class AnomalyEvent:
    """Structured operational excursion event with physical root-cause attribution."""
    timestamp: str
    plant_id: str
    actual_sec: float
    expected_sec: float
    z_score: float
    severity: str  # "Mild (2σ)" or "Severe (3σ)"
    failure_mode: str
    top_contributors: List[Dict[str, Any]]
    operator_guidance: str


class AnomalyRootCauseAnalyzer:
    """Detects DCS operational excursions and attributes root causes via local SHAP."""

    FAILURE_MODES = {
        "HEAT_STRESS": "Heat Stress / Surface Condenser Vacuum Loss",
        "COKING_RISK": "Reforming Under-Steaming / Coking Hazard (Low S/C)",
        "COMPRESSOR_SURGE": "Synthesis Loop High-Pressure / Compressor Overload",
        "CO2_SLIP": "CO₂ Absorption Breakthrough / Purge Gas Surge",
        "GAS_QUALITY": "HBJ Pipeline Natural Gas Heating Value Drop",
        "MULTIVARIATE_TRANSIENT": "Multi-Variable Plant Startup / Load Transient",
    }

    def __init__(self, model_artifact_path: Optional[Path | str] = None, logger: Optional[Any] = None) -> None:
        self.logger = logger or get_logger("anomaly_rootcause")
        self.paths = get_paths()
        self.figures_dir = self.paths["figures"]
        self.reports_dir = self.paths["reports"]

        # Load Champion Model
        models_dir = self.paths["models"]
        if model_artifact_path is None:
            lb_file = models_dir / "model_leaderboard.json"
            champ_file = "random_forest.joblib"
            if lb_file.exists():
                with open(lb_file, "r", encoding="utf-8") as f:
                    lb = json.load(f)
                if "xgboost" in lb.get("champion_model", "").lower():
                    champ_file = "xgboost_sec.joblib"
            model_artifact_path = models_dir / champ_file

        model_artifact_path = Path(model_artifact_path)
        artifact = joblib.load(model_artifact_path)
        self.model = artifact["model"]
        self.feature_names: List[str] = artifact["feature_names"]
        self.logger.info("Loaded surrogate model for root cause analysis from %s", model_artifact_path.name)

        self.explainer: Optional[shap.TreeExplainer] = None
        self.events: List[AnomalyEvent] = []

    def _classify_failure_mode(self, row: pd.Series, top_shap_features: List[Tuple[str, float]]) -> Tuple[str, str]:
        """Classify operational excursion into physical chemical engineering failure modes."""
        top_names = [f[0] for f in top_shap_features[:3]]
        
        # 1. Heat stress
        cw_temp = float(row.get("cw_inlet_temp", 30.0))
        amb_temp = float(row.get("ambient_temp", 32.0))
        if ("cw_inlet_temp" in top_names or "ambient_temp" in top_names) and (cw_temp > 33.5 or amb_temp > 42.0):
            return (
                self.FAILURE_MODES["HEAT_STRESS"],
                "Cooling water inlet elevated due to extreme Rajasthan ambient temperature; degraded surface condenser vacuum, increasing syngas compressor steam consumption."
            )

        # 2. Reforming under-steaming
        sc_ratio = float(row.get("steam_carbon_ratio", 3.05))
        if "steam_carbon_ratio" in top_names and sc_ratio < 2.92:
            return (
                self.FAILURE_MODES["COKING_RISK"],
                "Steam-to-Carbon ratio dropped near lower metallurgical limit; increased methane slip and risk of carbon coking on primary reformer nickel catalyst."
            )

        # 3. Compressor pressure surge
        loop_p = float(row.get("loop_pressure", 155.0))
        if "loop_pressure" in top_names and loop_p > 172.0:
            return (
                self.FAILURE_MODES["COMPRESSOR_SURGE"],
                "Synthesis loop pressure spike near safety relief boundary; elevated compressor power draw and purge valve throttling."
            )

        # 4. CO2 Breakthrough
        co2_slip = float(row.get("co2_slip_ppm", 600.0))
        if "co2_slip_ppm" in top_names or "co2_slip_ppm_rolling_mean_6h" in top_names:
            if co2_slip > 1400.0:
                return (
                    self.FAILURE_MODES["CO2_SLIP"],
                    "Solvent degradation in absorption column caused CO₂ slip surge; elevated methanator hydrogen consumption and synthesis loop purge rate."
                )

        # 5. Gas quality shock
        ng_cal = float(row.get("ng_calorific_value", 8600.0))
        if "ng_calorific_value" in top_names and ng_cal < 8480.0:
            return (
                self.FAILURE_MODES["GAS_QUALITY"],
                "HBJ natural gas heating value drop from upstream pipeline; required higher volumetric gas firing to sustain reforming equilibrium."
            )

        return (
            self.FAILURE_MODES["MULTIVARIATE_TRANSIENT"],
            "Coupled process perturbation across steam balance, synthesis loop recirculation, and ambient diurnal shift."
        )

    def detect_excursions(self, df: pd.DataFrame, z_threshold: float = 2.0) -> List[AnomalyEvent]:
        """Detect operational anomalies and decompose each excursion via SHAP."""
        self.logger.info("Initiating excursion detection across %d operational records (threshold=%.1fσ)...", len(df), z_threshold)

        # Compute per-plant rolling statistical baselines
        clean_df = df.sort_values("timestamp").dropna(subset=[TARGET_COLUMN]).copy().reset_index(drop=True)
        
        # Feature matrix
        X_mat = clean_df[self.feature_names].fillna(0)
        preds = self.model.predict(X_mat)
        clean_df["predicted_sec"] = preds
        clean_df["sec_residual"] = clean_df[TARGET_COLUMN] - preds

        # Calculate plant-specific mean and std of SEC
        plant_stats = clean_df.groupby("plant_id")[TARGET_COLUMN].agg(["mean", "std"]).to_dict("index")

        # Initialize SHAP explainer
        if self.explainer is None:
            self.logger.info("Initializing shap.TreeExplainer on surrogate model...")
            self.explainer = shap.TreeExplainer(self.model)

        excursion_events: List[AnomalyEvent] = []

        for idx, row in clean_df.iterrows():
            p_id = row["plant_id"]
            stats_p = plant_stats.get(p_id, {"mean": 5.15, "std": 0.05})
            actual_sec = float(row[TARGET_COLUMN])
            z = (actual_sec - stats_p["mean"]) / stats_p["std"] if stats_p["std"] > 0 else 0.0

            # Only flag high-SEC excursions (energy spikes, not super-efficient hours)
            if z >= z_threshold:
                severity = "Severe (3σ)" if z >= 3.0 else "Mild (2σ)"
                
                # Single-row SHAP computation
                x_single = X_mat.iloc[[idx]]
                shap_res = self.explainer(x_single)
                shap_vals = shap_res.values[0]

                # Top positive contributors (features pushing SEC up)
                ranked_idx = np.argsort(shap_vals)[::-1]
                top_contributors = []
                for fi in ranked_idx[:5]:
                    if shap_vals[fi] > 0:
                        top_contributors.append({
                            "feature": self.feature_names[fi],
                            "shap_impact": round(float(shap_vals[fi]), 4),
                            "measured_value": round(float(x_single.iloc[0, fi]), 2),
                        })

                top_tuples = [(c["feature"], c["shap_impact"]) for c in top_contributors]
                fail_mode, guidance = self._classify_failure_mode(row, top_tuples)

                event = AnomalyEvent(
                    timestamp=str(row["timestamp"]),
                    plant_id=p_id,
                    actual_sec=round(actual_sec, 4),
                    expected_sec=round(float(row["predicted_sec"]), 4),
                    z_score=round(float(z), 2),
                    severity=severity,
                    failure_mode=fail_mode,
                    top_contributors=top_contributors[:3],
                    operator_guidance=guidance,
                )
                excursion_events.append(event)

        self.events = excursion_events
        self.logger.info("Excursion detection complete: Identified %d anomalous operational hours.", len(excursion_events))
        return self.events

    # =========================================================================
    # Visualizations
    # =========================================================================

    def plot_anomaly_timeline(self, df: pd.DataFrame) -> Path:
        """Create publication-grade operational timeline highlighting excursion events."""
        apply_cfcl_theme()
        fig, ax = plt.subplots(figsize=(15, 6))

        clean_df = df.sort_values("timestamp").dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

        # Plot normal operational trajectory per plant
        for p_id in ["gadepan_1", "gadepan_2", "gadepan_3"]:
            mask = clean_df["plant_id"] == p_id
            p_data = clean_df[mask]
            ax.plot(
                p_data["timestamp"],
                p_data[TARGET_COLUMN],
                color=PLANT_COLORS[p_id],
                alpha=0.45,
                linewidth=1.0,
                label=PLANT_LABELS[p_id],
            )

        # Highlight Anomaly Events
        if self.events:
            ev_df = pd.DataFrame([asdict(e) for e in self.events])
            ev_df["timestamp"] = pd.to_datetime(ev_df["timestamp"])

            # 2σ mild excursions
            mild = ev_df[ev_df["severity"] == "Mild (2σ)"]
            if len(mild) > 0:
                ax.scatter(mild["timestamp"], mild["actual_sec"], color="#F39C12",
                           s=45, marker="^", edgecolor=CFCL_DARK, label=f"Mild Excursion 2σ (N={len(mild)})", zorder=5)

            # 3σ severe excursions
            severe = ev_df[ev_df["severity"] == "Severe (3σ)"]
            if len(severe) > 0:
                ax.scatter(severe["timestamp"], severe["actual_sec"], color="#E74C3C",
                           s=75, marker="X", edgecolor="black", label=f"Severe Incident 3σ (N={len(severe)})", zorder=6)

        ax.set_ylabel("Specific Energy Consumption (Gcal/MT urea)", fontsize=11, fontweight="bold")
        ax.set_title("CFCL Gadepan: Fleet SEC Operational Timeline & Excursion Event Detection",
                     fontsize=13, fontweight="bold", color=CFCL_BLUE, pad=12)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
        plt.tight_layout()

        out_path = self.figures_dir / "anomaly_timeline_scatter.png"
        save_figure(fig, out_path)
        plt.close(fig)
        self.logger.info("Saved anomaly timeline scatter plot to %s", out_path)
        return out_path

    def plot_worst_excursion_waterfall(self, df: pd.DataFrame) -> Path:
        """Generate high-resolution SHAP waterfall attribution for the peak excursion event."""
        apply_cfcl_theme()
        if not self.events:
            self.detect_excursions(df)

        # Find event with highest z-score
        worst_event = max(self.events, key=lambda e: e.z_score)
        
        # Isolate row in df
        target_ts = pd.to_datetime(worst_event.timestamp)
        match_row = df[(df["plant_id"] == worst_event.plant_id) & (pd.to_datetime(df["timestamp"]) == target_ts)]
        
        if len(match_row) == 0:
            match_row = df.tail(1)

        X_row = match_row[self.feature_names].fillna(0)
        shap_res = self.explainer(X_row)

        fig, ax = plt.subplots(figsize=(12, 7))
        shap.plots.waterfall(shap_res[0], max_display=12, show=False)

        plt.title(
            f"Excursion Root-Cause Decomposition: {worst_event.plant_id.upper()} ({worst_event.timestamp})\n"
            f"Measured SEC: {worst_event.actual_sec:.3f} Gcal/MT (Z-Score: +{worst_event.z_score}σ) | Mode: {worst_event.failure_mode}",
            fontsize=12,
            fontweight="bold",
            color="#C0392B",
            pad=14,
        )
        plt.tight_layout()

        out_path = self.figures_dir / "anomaly_excursion_waterfall.png"
        save_figure(plt.gcf(), out_path)
        plt.close("all")
        self.logger.info("Saved peak excursion waterfall plot to %s", out_path)
        return out_path

    def export_report_json(self) -> Path:
        """Export comprehensive incident audit log to reports."""
        events_dicts = [asdict(e) for e in self.events]
        
        # Summary counts
        mild_count = sum(1 for e in self.events if e.severity == "Mild (2σ)")
        severe_count = sum(1 for e in self.events if e.severity == "Severe (3σ)")
        
        mode_counts: Dict[str, int] = {}
        for e in self.events:
            mode_counts[e.failure_mode] = mode_counts.get(e.failure_mode, 0) + 1

        plant_counts: Dict[str, int] = {}
        for e in self.events:
            plant_counts[e.plant_id] = plant_counts.get(e.plant_id, 0) + 1

        report = {
            "detection_timestamp": pd.Timestamp.now().isoformat(),
            "total_excursion_hours": len(self.events),
            "severity_breakdown": {
                "mild_2sigma": mild_count,
                "severe_3sigma": severe_count,
            },
            "failure_mode_breakdown": mode_counts,
            "plant_breakdown": plant_counts,
            "recent_incidents": events_dicts[-15:] if len(events_dicts) > 15 else events_dicts,
        }

        out_path = self.reports_dir / "anomaly_rootcause_report.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        self.logger.info("Saved anomaly root-cause report to %s", out_path)
        return out_path

    def run_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Execute complete excursion detection, attribution, and reporting."""
        self.logger.info("Initiating full Anomaly Root-Cause Analysis pipeline...")
        self.detect_excursions(df)
        self.plot_anomaly_timeline(df)
        self.plot_worst_excursion_waterfall(df)
        self.export_report_json()
        return {
            "status": "success",
            "total_excursions": len(self.events),
            "severe_incidents": sum(1 for e in self.events if e.severity == "Severe (3σ)"),
        }


def main() -> None:
    """Run CLI."""
    paths = get_paths()
    df = pd.read_parquet(paths["processed_data"] / "plant_features.parquet")
    analyzer = AnomalyRootCauseAnalyzer()
    analyzer.run_analysis(df)


if __name__ == "__main__":
    main()
