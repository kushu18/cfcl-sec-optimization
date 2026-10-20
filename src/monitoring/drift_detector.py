"""DCS Telemetry Concept & Covariate Drift Monitor for CFCL Gadepan.

Implements two-sample Kolmogorov-Smirnov (KS) hypothesis testing to track
statistical distribution shift between reference plant baseline and live telemetry:
1. Multi-feature KS testing across core process setpoints, ambient conditions,
   feed gas quality, and catalyst aging.
2. Enterprise Drift Severity Classification:
   - NORMAL: 0 drifted features (DCS telemetry stable)
   - LOW: 1 drifted feature (Sensor zero/span calibration advisory)
   - MODERATE: 2 drifted features (Supervisory inspection recommended)
   - HIGH (CRITICAL): ≥ 3 drifted features (Model retraining & baseline reset triggered)
3. Generates publication-quality drift heatmaps, distribution overlay plots,
   and structured audit JSON reports.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

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


@dataclass
class FeatureDriftResult:
    """Statistical test outcome for an individual DCS telemetry tag."""
    feature_name: str
    ks_statistic: float
    p_value: float
    is_drifted: bool
    baseline_mean: float
    monitoring_mean: float
    percentage_mean_shift: float
    severity_flag: str  # "STABLE", "DRIFT_DETECTED", "CRITICAL_DRIFT"


class ConceptDriftDetector:
    """Enterprise DCS Covariate & Concept Drift Monitoring Engine."""

    # Key physical tags monitored continuously
    MONITORED_FEATURES: List[str] = [
        "steam_carbon_ratio",
        "reformer_outlet_temp",
        "loop_pressure",
        "ambient_temp",
        "ambient_humidity",
        "cw_inlet_temp",
        "ng_calorific_value",
        "co2_slip_ppm",
        "catalyst_age_days",
        TARGET_COLUMN,
    ]

    def __init__(self, significance_level: float = 0.05, logger: Optional[Any] = None) -> None:
        self.logger = logger or get_logger("drift_detector")
        self.paths = get_paths()
        self.figures_dir = self.paths["figures"]
        self.reports_dir = self.paths["reports"]
        self.alpha = significance_level

        self.drift_results: List[FeatureDriftResult] = []
        self.overall_severity: str = "NORMAL"
        self.ref_data: Optional[pd.DataFrame] = None
        self.curr_data: Optional[pd.DataFrame] = None

    def partition_baseline_and_monitoring(
        self,
        df: pd.DataFrame,
        baseline_fraction: float = 0.35,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split chronological telemetry into reference baseline and active monitoring windows."""
        clean_df = df.sort_values("timestamp").dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
        split_idx = int(len(clean_df) * baseline_fraction)

        self.ref_data = clean_df.iloc[:split_idx].copy().reset_index(drop=True)
        self.curr_data = clean_df.iloc[split_idx:].copy().reset_index(drop=True)

        self.logger.info(
            "Partitioned telemetry: Reference Window (%d records, %s to %s) | Monitoring Window (%d records, %s to %s)",
            len(self.ref_data),
            self.ref_data["timestamp"].min(),
            self.ref_data["timestamp"].max(),
            len(self.curr_data),
            self.curr_data["timestamp"].min(),
            self.curr_data["timestamp"].max(),
        )
        return self.ref_data, self.curr_data

    def run_ks_tests(self) -> List[FeatureDriftResult]:
        """Execute two-sample Kolmogorov-Smirnov test for each monitored process tag."""
        if self.ref_data is None or self.curr_data is None:
            raise ValueError("Reference and monitoring windows must be partitioned first.")

        self.logger.info("Executing two-sample KS hypothesis testing (alpha=%.3f)...", self.alpha)
        results: List[FeatureDriftResult] = []
        drifted_count = 0

        for feat in self.MONITORED_FEATURES:
            if feat not in self.ref_data.columns or feat not in self.curr_data.columns:
                continue

            ref_series = self.ref_data[feat].dropna().values
            curr_series = self.curr_data[feat].dropna().values

            ks_stat, p_val = stats.ks_2samp(ref_series, curr_series)
            is_drift = bool(p_val < self.alpha)
            
            m_ref = float(np.mean(ref_series))
            m_curr = float(np.mean(curr_series))
            shift_pct = ((m_curr - m_ref) / m_ref * 100.0) if m_ref != 0 else 0.0

            if is_drift:
                drifted_count += 1
                flag = "CRITICAL_DRIFT" if p_val < 0.001 else "DRIFT_DETECTED"
            else:
                flag = "STABLE"

            res = FeatureDriftResult(
                feature_name=feat,
                ks_statistic=float(round(ks_stat, 4)),
                p_value=float(round(p_val, 6)),
                is_drifted=is_drift,
                baseline_mean=float(round(m_ref, 2)),
                monitoring_mean=float(round(m_curr, 2)),
                percentage_mean_shift=float(round(shift_pct, 2)),
                severity_flag=flag,
            )
            results.append(res)
            self.logger.info("Feature %-22s | KS Stat=%.4f | p-val=%.4e | Status: %s",
                             feat, ks_stat, p_val, flag)

        self.drift_results = results

        # Determine Fleet-Wide Severity
        if drifted_count == 0:
            self.overall_severity = "NORMAL"
        elif drifted_count == 1:
            self.overall_severity = "LOW_SEVERITY"
        elif drifted_count == 2:
            self.overall_severity = "MODERATE_SEVERITY"
        else:
            self.overall_severity = "HIGH_SEVERITY_CRITICAL"

        self.logger.info("Fleet Drift Severity Assessment: %s (%d/%d tags drifted)",
                         self.overall_severity, drifted_count, len(results))
        return self.drift_results

    # =========================================================================
    # Visualizations
    # =========================================================================

    def plot_drift_heatmap(self, df: pd.DataFrame) -> Path:
        """Create temporal monthly drift heatmap comparing each month against baseline."""
        apply_cfcl_theme()
        clean_df = df.sort_values("timestamp").copy()
        clean_df["month_label"] = pd.to_datetime(clean_df["timestamp"]).dt.strftime("%b %Y")

        months = list(clean_df["month_label"].unique())
        if len(months) < 2:
            months = ["Baseline Window", "Monitoring Window"]

        ref_month_data = clean_df[clean_df["month_label"] == months[0]]
        features = [f for f in self.MONITORED_FEATURES if f in clean_df.columns]

        ks_matrix = np.zeros((len(features), len(months)))

        for j, m_str in enumerate(months):
            curr_m_data = clean_df[clean_df["month_label"] == m_str]
            for i, feat in enumerate(features):
                if len(curr_m_data) > 10 and len(ref_month_data) > 10:
                    stat_val, _ = stats.ks_2samp(ref_month_data[feat].dropna(), curr_m_data[feat].dropna())
                    ks_matrix[i, j] = stat_val
                else:
                    ks_matrix[i, j] = 0.0

        fig, ax = plt.subplots(figsize=(12, 7))
        cax = ax.imshow(ks_matrix, cmap="YlOrRd", aspect="auto", vmin=0.0, vmax=0.8)

        # Labels
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels(months, fontsize=10, fontweight="bold")
        ax.set_yticks(range(len(features)))
        display_feats = [f.replace("_", " ").title() for f in features]
        ax.set_yticklabels(display_feats, fontsize=10, fontweight="bold")

        # Add text values inside cells
        for i in range(len(features)):
            for j in range(len(months)):
                val = ks_matrix[i, j]
                text_col = "white" if val > 0.45 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_col, fontsize=9, fontweight="bold")

        cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Kolmogorov-Smirnov Statistic (D)", fontsize=10, fontweight="bold")

        ax.set_title("CFCL Gadepan: Multi-Month Covariate Drift Heatmap (vs. Jan 2026 Baseline)",
                     fontsize=13, fontweight="bold", color=CFCL_BLUE, pad=12)
        plt.tight_layout()

        out_path = self.figures_dir / "sensor_drift_heatmap.png"
        save_figure(fig, out_path)
        plt.close(fig)
        self.logger.info("Saved sensor drift heatmap to %s", out_path)
        return out_path

    def plot_distribution_shifts(self) -> Path:
        """Create 4-panel distribution comparison for key drifted physical tags."""
        apply_cfcl_theme()
        if self.ref_data is None or self.curr_data is None:
            raise ValueError("Data not partitioned.")

        key_tags = ["ambient_temp", "cw_inlet_temp", "catalyst_age_days", TARGET_COLUMN]
        display_names = [
            "Ambient Temperature (°C)",
            "Cooling Water Inlet Temp (°C)",
            "Catalyst Age (Days)",
            "Specific Energy Consumption (Gcal/MT)",
        ]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, tag in enumerate(key_tags):
            ax = axes[idx]
            ref_vals = self.ref_data[tag].dropna().values
            curr_vals = self.curr_data[tag].dropna().values

            # Overlaid KDE plots
            kde_ref = stats.gaussian_kde(ref_vals)
            kde_curr = stats.gaussian_kde(curr_vals)

            min_val = min(min(ref_vals), min(curr_vals))
            max_val = max(max(ref_vals), max(curr_vals))
            x_eval = np.linspace(min_val, max_val, 150)

            ax.plot(x_eval, kde_ref(x_eval), color=CFCL_BLUE, linewidth=2.2, label="Reference Baseline (Winter)")
            ax.fill_between(x_eval, 0, kde_ref(x_eval), color=CFCL_BLUE, alpha=0.15)

            ax.plot(x_eval, kde_curr(x_eval), color="#E74C3C", linewidth=2.2, linestyle="--", label="Monitoring Window (Summer)")
            ax.fill_between(x_eval, 0, kde_curr(x_eval), color="#E74C3C", alpha=0.15)

            # Find KS statistic for this tag
            ks_res = next((r for r in self.drift_results if r.feature_name == tag), None)
            stat_str = f"KS D={ks_res.ks_statistic:.3f} (p={ks_res.p_value:.1e})" if ks_res else ""

            ax.set_title(f"{display_names[idx]}\n{stat_str}", fontsize=11, fontweight="bold", color=CFCL_DARK)
            ax.set_ylabel("Probability Density", fontsize=9)
            ax.grid(True, linestyle=":", alpha=0.6)
            if idx == 0:
                ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

        plt.suptitle("CFCL Gadepan: Reference Baseline vs. Operational Drift Density Shift",
                     fontsize=14, fontweight="bold", color=CFCL_BLUE, y=0.995)
        plt.tight_layout()

        out_path = self.figures_dir / "ks_drift_distributions.png"
        save_figure(fig, out_path)
        plt.close(fig)
        self.logger.info("Saved KS drift distribution comparison to %s", out_path)
        return out_path

    def export_report_json(self) -> Path:
        """Export comprehensive drift monitoring report to reports directory."""
        results_dicts = [asdict(r) for r in self.drift_results]
        drifted_tags = [r.feature_name for r in self.drift_results if r.is_drifted]

        guidance_map = {
            "NORMAL": "All physical process variables and environmental conditions align with reference baseline. No calibration required.",
            "LOW_SEVERITY": "Single variable distribution shift detected. Issue routine zero/span calibration advisory to DCS instrumentation team.",
            "MODERATE_SEVERITY": "Two critical process variables drifted. Inspect cooling tower approach temperature and gas chromatograph calibration.",
            "HIGH_SEVERITY_CRITICAL": "Severe multi-variable distribution shift detected (Summer heatwave / catalyst aging). Trigger automated ML model retraining and update steady-state optimization baseline.",
        }

        report = {
            "monitoring_timestamp": pd.Timestamp.now().isoformat(),
            "overall_severity": self.overall_severity,
            "significance_level": self.alpha,
            "monitored_tags_count": len(self.drift_results),
            "drifted_tags_count": len(drifted_tags),
            "drifted_tags": drifted_tags,
            "recommended_dcs_action": guidance_map.get(self.overall_severity, "Inspect DCS telemetry."),
            "feature_diagnostics": results_dicts,
        }

        out_path = self.reports_dir / "drift_monitoring_report.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        self.logger.info("Saved drift monitoring report to %s", out_path)
        return out_path

    def run_monitoring_cycle(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Execute full telemetry drift detection and reporting cycle."""
        self.logger.info("Initiating full DCS Concept Drift Monitoring cycle...")
        self.partition_baseline_and_monitoring(df)
        self.run_ks_tests()
        self.plot_drift_heatmap(df)
        self.plot_distribution_shifts()
        self.export_report_json()
        return {
            "status": "success",
            "severity": self.overall_severity,
            "drifted_count": sum(1 for r in self.drift_results if r.is_drifted),
        }


def main() -> None:
    """Run CLI."""
    paths = get_paths()
    df = pd.read_parquet(paths["processed_data"] / "plant_features.parquet")
    detector = ConceptDriftDetector()
    detector.run_monitoring_cycle(df)


if __name__ == "__main__":
    main()
