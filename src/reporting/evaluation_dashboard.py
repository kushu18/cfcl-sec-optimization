"""Interactive Executive Model Diagnostic Dashboard for CFCL SEC Optimization.

Generates a standalone, publication-grade HTML report combining:
- Executive KPI summary cards (Fleet Capacity, Champion Metrics, DCS Tolerance Bands)
- Interactive tabs for Parity Analysis, Residual Diagnostics, Fleet Generalization
- SHAP Game-Theoretic Feature Attribution & Rajasthan Weather Impact
- Chemical Process Variable Elasticity (Manipulated vs. Disturbance Sensitivities)
- Embedded high-resolution publication plots and structured benchmark tables
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.config_loader import get_paths, get_project_root
from src.utils.logger import get_logger


class EvaluationDashboardGenerator:
    """Generates a standalone, self-contained interactive executive HTML dashboard."""

    def __init__(self, logger: Optional[Any] = None) -> None:
        self.logger = logger or get_logger("evaluation_dashboard")
        self.paths = get_paths()
        self.figures_dir = self.paths["figures"]
        self.models_dir = self.paths["models"]
        self.reports_dir = self.paths["reports"]

    def _encode_image(self, filename: str) -> str:
        """Encode a figure as a base64 data URI for self-contained HTML rendering."""
        img_path = self.figures_dir / filename
        if not img_path.exists():
            # Try without extension
            img_path = self.figures_dir / f"{filename}.png"
        if img_path.exists():
            data = img_path.read_bytes()
            b64 = base64.b64encode(data).decode("utf-8")
            return f"data:image/png;base64,{b64}"
        return ""

    def generate_html(self) -> Path:
        """Build and save the self-contained HTML executive dashboard."""
        self.logger.info("Generating interactive executive model evaluation dashboard...")

        # Load Leaderboard
        lb_path = self.models_dir / "model_leaderboard.json"
        with open(lb_path, "r", encoding="utf-8") as f:
            lb = json.load(f)

        # Load Elasticity Report
        fi_path = self.reports_dir / "feature_importance_report.json"
        with open(fi_path, "r", encoding="utf-8") as f:
            fir = json.load(f)

        # Load SHAP Summary
        shap_path = self.reports_dir / "shap_summary.json"
        with open(shap_path, "r", encoding="utf-8") as f:
            shap_data = json.load(f)

        champ_name = lb["champion_model"].replace("_", " ")
        champ_metrics = lb["champion_metrics"]

        # Encode key figures
        img_leaderboard = self._encode_image("model_leaderboard_comparison.png")
        img_parity = self._encode_image("actual_vs_predicted_all_models.png")
        img_residuals = self._encode_image("residual_diagnostics.png")
        img_fleet = self._encode_image("within_plant_performance.png")
        img_beeswarm = self._encode_image("shap_summary_beeswarm.png")
        img_shap_bar = self._encode_image("shap_feature_importance_bar.png")
        img_worst_waterfall = self._encode_image("shap_waterfall_worst_hour.png")
        img_dep_sr = self._encode_image("shap_dependence_steam_reformer.png")
        img_dep_cw = self._encode_image("shap_dependence_ambient_cw.png")
        img_elasticity = self._encode_image("process_variable_elasticity.png")

        # Build Elasticity Table Rows
        elasticity_rows = ""
        for row in fir.get("process_variable_elasticity", []):
            is_mv = "Manipulated" in row["category"]
            badge_class = "badge-mv" if is_mv else "badge-dv"
            badge_text = "Manipulated (MV)" if is_mv else "Disturbance (DV)"
            delta_val = row["delta_sec_gcal_per_mt"]
            delta_color = "#27AE60" if delta_val < 0 else "#E74C3C"

            elasticity_rows += f"""
            <tr>
                <td><strong>{row['variable'].replace('_', ' ').title()}</strong></td>
                <td><span class="badge {badge_class}">{badge_text}</span></td>
                <td><code>{row['step_change']}</code></td>
                <td>{row['mean_value']}</td>
                <td style="color: {delta_color}; font-weight: bold;">{delta_val:+.5f} Gcal/MT</td>
                <td>{row['percent_sec_impact']:+.3f}%</td>
                <td style="font-size: 13px;">{row['operational_guidance']}</td>
            </tr>
            """

        # Build Model Comparison Rows
        model_rows = ""
        for m_name, m_data in lb["models"].items():
            is_champ = m_name == lb["champion_model"]
            row_style = "background-color: #E8F8F5; font-weight: bold;" if is_champ else ""
            status_badge = '<span class="badge badge-champ">🏆 Champion</span>' if is_champ else '<span class="badge badge-runner">Benchmarked</span>'
            model_rows += f"""
            <tr style="{row_style}">
                <td>{m_name.replace('_', ' ')} {status_badge}</td>
                <td>{m_data['r2_score']:.4f}</td>
                <td>{m_data['rmse']:.4f}</td>
                <td>{m_data['mae']:.4f}</td>
                <td>{m_data['within_1pct_dcs_band']:.1f}%</td>
                <td>{m_data['within_2pct_dcs_band']:.1f}%</td>
                <td>{m_data['durbin_watson']:.3f}</td>
                <td>{m_data['heteroscedasticity_corr']:+.3f}</td>
            </tr>
            """

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CFCL Gadepan - SEC Optimization Engine Executive Diagnostic Dashboard</title>
    <style>
        :root {{
            --cfcl-blue: #1B4F72;
            --cfcl-light-blue: #2E86C1;
            --cfcl-dark: #2C3E50;
            --cfcl-bg: #F4F6F9;
            --cfcl-card: #FFFFFF;
            --success: #27AE60;
            --warning: #F39C12;
            --danger: #E74C3C;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--cfcl-bg);
            color: #333333;
            margin: 0;
            padding: 0;
            line-height: 1.5;
        }}
        header {{
            background: linear-gradient(135deg, var(--cfcl-blue) 0%, #154360 100%);
            color: white;
            padding: 24px 36px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        .header-content {{
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .header-title h1 {{
            margin: 0;
            font-size: 26px;
            letter-spacing: -0.5px;
        }}
        .header-title p {{
            margin: 6px 0 0 0;
            font-size: 14px;
            opacity: 0.9;
        }}
        .header-badge {{
            background-color: rgba(255,255,255,0.15);
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 13px;
            border: 1px solid rgba(255,255,255,0.25);
        }}
        .container {{
            max-width: 1400px;
            margin: 28px auto;
            padding: 0 24px;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 20px;
            margin-bottom: 28px;
        }}
        .kpi-card {{
            background-color: var(--cfcl-card);
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            border-left: 5px solid var(--cfcl-light-blue);
        }}
        .kpi-card.champ {{
            border-left-color: var(--success);
        }}
        .kpi-label {{
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            color: #7F8C8D;
            letter-spacing: 0.5px;
        }}
        .kpi-value {{
            font-size: 28px;
            font-weight: 800;
            color: var(--cfcl-dark);
            margin: 8px 0;
        }}
        .kpi-subtext {{
            font-size: 13px;
            color: #555;
        }}
        .card {{
            background-color: var(--cfcl-card);
            border-radius: 8px;
            padding: 24px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            margin-bottom: 28px;
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #EAEDED;
            padding-bottom: 14px;
            margin-bottom: 20px;
        }}
        .card-header h2 {{
            margin: 0;
            font-size: 18px;
            color: var(--cfcl-blue);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        th, td {{
            padding: 12px 14px;
            text-align: left;
            border-bottom: 1px solid #ECEFF1;
        }}
        th {{
            background-color: #F8F9F9;
            color: var(--cfcl-dark);
            font-weight: 700;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.4px;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 700;
        }}
        .badge-champ {{ background-color: #D4EFDF; color: #196F3D; }}
        .badge-runner {{ background-color: #EBF5FB; color: #21618C; }}
        .badge-mv {{ background-color: #D5F5E3; color: #1E8449; }}
        .badge-dv {{ background-color: #FCF3CF; color: #B7950B; }}
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        img.responsive-fig {{
            width: 100%;
            height: auto;
            border-radius: 6px;
            border: 1px solid #E5E7E9;
        }}
        footer {{
            text-align: center;
            padding: 24px;
            color: #7F8C8D;
            font-size: 13px;
        }}
    </style>
</head>
<body>

<header>
    <div class="header-content">
        <div class="header-title">
            <h1>CFCL Gadepan - SEC Optimization Engine</h1>
            <p>Model Evaluation, SHAP Attribution & Process Sensitivity Diagnostic Report</p>
        </div>
        <div class="header-badge">
            <strong>Facility:</strong> Gadepan (Kota, Rajasthan) &nbsp;|&nbsp; <strong>Fleet Capacity:</strong> 5,390 MTPD Urea
        </div>
    </div>
</header>

<div class="container">

    <!-- KPI Summary Cards -->
    <div class="kpi-grid">
        <div class="kpi-card champ">
            <div class="kpi-label">Production Champion Model</div>
            <div class="kpi-value">{champ_name}</div>
            <div class="kpi-subtext">Holdout R²: <strong>{champ_metrics['r2_score']:.4f}</strong> | RMSE: <strong>{champ_metrics['rmse']:.4f} Gcal/MT</strong></div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Industrial DCS Precision</div>
            <div class="kpi-value">{champ_metrics['within_1pct_dcs_band']:.1f}%</div>
            <div class="kpi-subtext">Within ±1.0% tolerance band (<strong>{champ_metrics['within_2pct_dcs_band']:.1f}%</strong> within ±2%)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Residual Autocorrelation</div>
            <div class="kpi-value">{champ_metrics['durbin_watson']:.3f}</div>
            <div class="kpi-subtext">Durbin-Watson statistic (Ideal = 2.0; Low lag bias)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">95% DCS Prediction Interval</div>
            <div class="kpi-value">±{champ_metrics['prediction_interval_95']:.4f}</div>
            <div class="kpi-subtext">Gcal/MT urea (Heteroscedasticity r = {champ_metrics['heteroscedasticity_corr']:+.3f})</div>
        </div>
    </div>

    <!-- Model Comparison Table -->
    <div class="card">
        <div class="card-header">
            <h2>1. Unified Multi-Model Evaluation Leaderboard (2,270 Holdout Records)</h2>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Model Family</th>
                    <th>Goodness of Fit (R²)</th>
                    <th>RMSE (Gcal/MT)</th>
                    <th>MAE (Gcal/MT)</th>
                    <th>Within ±1% DCS Band</th>
                    <th>Within ±2% DCS Band</th>
                    <th>Durbin-Watson</th>
                    <th>Heteroscedasticity Corr</th>
                </tr>
            </thead>
            <tbody>
                {model_rows}
            </tbody>
        </table>
    </div>

    <!-- Parity and Leaderboard Plots -->
    <div class="grid-2">
        <div class="card">
            <div class="card-header">
                <h2>2. Goodness of Fit & Parity Envelopes (±2% Tolerance)</h2>
            </div>
            <img class="responsive-fig" src="{img_parity}" alt="Actual vs Predicted Parity Plot">
        </div>
        <div class="card">
            <div class="card-header">
                <h2>3. Cross-Model Metric Comparison (R², RMSE, MAE)</h2>
            </div>
            <img class="responsive-fig" src="{img_leaderboard}" alt="Model Leaderboard Comparison">
        </div>
    </div>

    <!-- Fleet Generalization and Residual Diagnostics -->
    <div class="grid-2">
        <div class="card">
            <div class="card-header">
                <h2>4. Operating Unit Breakdown (Gadepan-I, II, and III)</h2>
            </div>
            <img class="responsive-fig" src="{img_fleet}" alt="Within-Plant Performance">
        </div>
        <div class="card">
            <div class="card-header">
                <h2>5. Residual Diagnostics & Gaussian Distribution</h2>
            </div>
            <img class="responsive-fig" src="{img_residuals}" alt="Residual Diagnostics">
        </div>
    </div>

    <!-- Process Variable Elasticity Table -->
    <div class="card">
        <div class="card-header">
            <h2>6. Operational Process Variable Elasticity & Sensitivity Table</h2>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Variable</th>
                    <th>Control Type</th>
                    <th>Perturbation Step</th>
                    <th>Baseline Mean</th>
                    <th>Marginal Impact (ΔSEC)</th>
                    <th>% SEC Change</th>
                    <th>Operational Engineering Guidance</th>
                </tr>
            </thead>
            <tbody>
                {elasticity_rows}
            </tbody>
        </table>
    </div>

    <!-- Elasticity & Permutation Plots -->
    <div class="grid-2">
        <div class="card">
            <div class="card-header">
                <h2>7. Operational Setpoint Sensitivity (MVs vs. DVs)</h2>
            </div>
            <img class="responsive-fig" src="{img_elasticity}" alt="Process Variable Elasticity">
        </div>
        <div class="card">
            <div class="card-header">
                <h2>8. SHAP Game-Theoretic Feature Attribution (Top 15)</h2>
            </div>
            <img class="responsive-fig" src="{img_shap_bar}" alt="SHAP Feature Importance Bar">
        </div>
    </div>

    <!-- SHAP Thermodynamic Dependence -->
    <div class="grid-2">
        <div class="card">
            <div class="card-header">
                <h2>9. Thermodynamic Coupling: Steam/Carbon vs. Reformer Temp</h2>
            </div>
            <img class="responsive-fig" src="{img_dep_sr}" alt="SHAP Steam Reformer Interaction">
        </div>
        <div class="card">
            <div class="card-header">
                <h2>10. Rajasthan Heatwave Coupling: Ambient vs. Cooling Water</h2>
            </div>
            <img class="responsive-fig" src="{img_dep_cw}" alt="SHAP Ambient CW Interaction">
        </div>
    </div>

</div>

<footer>
    <p>Chambal Fertilisers and Chemicals Limited (CFCL) — Gadepan Complex (Kota, Rajasthan)<br>
    Specific Energy Consumption (SEC) Optimization Engine &copy; 2026. Confidential — Internal DCS Technical Report.</p>
</footer>

</body>
</html>
"""
        out_path = self.reports_dir / "model_evaluation_dashboard.html"
        out_path.write_text(html_content, encoding="utf-8")
        self.logger.info("Saved interactive executive dashboard to %s (%d KB)", out_path, len(html_content) // 1024)
        return out_path


def main() -> None:
    """Generate executive dashboard CLI."""
    generator = EvaluationDashboardGenerator()
    generator.generate_html()


if __name__ == "__main__":
    main()
