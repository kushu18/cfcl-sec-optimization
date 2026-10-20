"""Interactive Executive Anomaly & Drift Monitoring Dashboard for CFCL Gadepan.

Generates a standalone, publication-grade HTML report combining:
- Executive Alert Banners & Severity Badges (Low/Moderate/Critical)
- Statistical Excursion Timeline (2σ / 3σ incident identification)
- Local SHAP Root-Cause Failure-Mode Breakdown
- Multi-Month Kolmogorov-Smirnov (KS) Covariate Drift Heatmap
- Baseline vs. Monitoring Density Shift Curves
- Embedded high-resolution publication figures and incident tables
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.config_loader import get_paths, get_project_root
from src.utils.logger import get_logger


class MonitoringDashboardGenerator:
    """Generates an executive-ready HTML dashboard for Part 8 Anomaly & Drift."""

    def __init__(self, logger: Optional[Any] = None) -> None:
        self.logger = logger or get_logger("monitoring_dashboard")
        self.paths = get_paths()
        self.figures_dir = self.paths["figures"]
        self.reports_dir = self.paths["reports"]

    def _encode_image(self, filename: str) -> str:
        """Encode figure as base64 data URI."""
        img_path = self.figures_dir / filename
        if not img_path.exists():
            img_path = self.figures_dir / f"{filename}.png"
        if img_path.exists():
            data = img_path.read_bytes()
            b64 = base64.b64encode(data).decode("utf-8")
            return f"data:image/png;base64,{b64}"
        return ""

    def generate_html(self) -> Path:
        """Build and save self-contained HTML executive monitoring dashboard."""
        self.logger.info("Generating interactive executive anomaly & drift dashboard...")

        # Load Anomaly Report
        anom_path = self.reports_dir / "anomaly_rootcause_report.json"
        with open(anom_path, "r", encoding="utf-8") as f:
            anom_rep = json.load(f)

        # Load Drift Report
        drift_path = self.reports_dir / "drift_monitoring_report.json"
        with open(drift_path, "r", encoding="utf-8") as f:
            drift_rep = json.load(f)

        # Encode figures
        img_timeline = self._encode_image("anomaly_timeline_scatter.png")
        img_waterfall = self._encode_image("anomaly_excursion_waterfall.png")
        img_heatmap = self._encode_image("sensor_drift_heatmap.png")
        img_distributions = self._encode_image("ks_drift_distributions.png")

        # Build Incident Table Rows
        incident_rows = ""
        for inc in anom_rep.get("recent_incidents", []):
            is_severe = "3σ" in inc["severity"]
            sev_badge = '<span class="badge badge-severe">Severe (3σ)</span>' if is_severe else '<span class="badge badge-mild">Mild (2σ)</span>'
            
            top_drivers = ", ".join([f"{c['feature'].replace('_', ' ')} (+{c['shap_impact']:.3f})" for c in inc.get("top_contributors", [])])

            incident_rows += f"""
            <tr>
                <td><code>{inc['timestamp']}</code></td>
                <td><strong>{inc['plant_id'].upper()}</strong></td>
                <td>{inc['actual_sec']:.3f} Gcal/MT</td>
                <td>+{inc['z_score']:.2f}σ</td>
                <td>{sev_badge}</td>
                <td><strong>{inc['failure_mode']}</strong></td>
                <td style="font-size: 13px;">{top_drivers}</td>
                <td style="font-size: 13px; color: #555;">{inc['operator_guidance']}</td>
            </tr>
            """

        # Build Drift Diagnostics Table Rows
        drift_rows = ""
        for tag in drift_rep.get("feature_diagnostics", []):
            is_d = tag["is_drifted"]
            flag_badge = f'<span class="badge badge-severe">{tag["severity_flag"]}</span>' if is_d else '<span class="badge badge-stable">STABLE</span>'
            p_val_str = f"{tag['p_value']:.2e}" if tag['p_value'] < 0.001 else f"{tag['p_value']:.4f}"

            drift_rows += f"""
            <tr>
                <td><strong>{tag['feature_name'].replace('_', ' ').title()}</strong></td>
                <td><code>{tag['ks_statistic']:.4f}</code></td>
                <td>{p_val_str}</td>
                <td>{tag['baseline_mean']}</td>
                <td>{tag['monitoring_mean']}</td>
                <td style="font-weight: bold; color: {'#E74C3C' if abs(tag['percentage_mean_shift']) > 5 else '#2C3E50'};">{tag['percentage_mean_shift']:+.2f}%</td>
                <td>{flag_badge}</td>
            </tr>
            """

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CFCL Gadepan - Anomaly Root-Cause & Concept Drift Executive Monitor</title>
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
        .kpi-card.warning {{
            border-left-color: var(--warning);
        }}
        .kpi-card.danger {{
            border-left-color: var(--danger);
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
        .badge-stable {{ background-color: #D4EFDF; color: #196F3D; }}
        .badge-mild {{ background-color: #FCF3CF; color: #B7950B; }}
        .badge-severe {{ background-color: #FADBD8; color: #922B21; }}
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
            <h1>CFCL Gadepan - Anomaly Root-Cause & Concept Drift Monitor</h1>
            <p>Automated Operational Excursion Diagnostics & Kolmogorov-Smirnov Statistical Testing</p>
        </div>
        <div class="header-badge">
            <strong>Facility:</strong> Gadepan (Kota, Rajasthan) &nbsp;|&nbsp; <strong>Fleet Capacity:</strong> 5,390 MTPD Urea
        </div>
    </div>
</header>

<div class="container">

    <!-- KPI Summary Cards -->
    <div class="kpi-grid">
        <div class="kpi-card danger">
            <div class="kpi-label">DCS Drift Severity Status</div>
            <div class="kpi-value">{drift_rep['overall_severity'].replace('_', ' ')}</div>
            <div class="kpi-subtext">{drift_rep['drifted_tags_count']} of {drift_rep['monitored_tags_count']} process tags statistically shifted (p &lt; 0.05)</div>
        </div>
        <div class="kpi-card warning">
            <div class="kpi-label">Operational Excursions (2σ)</div>
            <div class="kpi-value">{anom_rep['total_excursion_hours']} Hours</div>
            <div class="kpi-subtext">{anom_rep['severity_breakdown']['severe_3sigma']} critical 3σ events detected across fleet</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Primary Failure Mode</div>
            <div class="kpi-value" style="font-size: 20px; margin-top: 12px;">Heat Stress / Condenser</div>
            <div class="kpi-subtext">Triggered by Kota May-June ambient heatwave (>42°C)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Recommended DCS Action</div>
            <div class="kpi-value" style="font-size: 16px; margin-top: 14px; color: #C0392B;">Model Retraining Alert</div>
            <div class="kpi-subtext">{drift_rep['recommended_dcs_action'][:75]}...</div>
        </div>
    </div>

    <!-- Anomaly Timeline & Excursion Waterfall -->
    <div class="grid-2">
        <div class="card">
            <div class="card-header">
                <h2>1. Fleet Operational Timeline & Excursion Identification</h2>
            </div>
            <img class="responsive-fig" src="{img_timeline}" alt="Anomaly Timeline Scatter">
        </div>
        <div class="card">
            <div class="card-header">
                <h2>2. Peak Excursion Root-Cause Decomposition (SHAP)</h2>
            </div>
            <img class="responsive-fig" src="{img_waterfall}" alt="Anomaly Excursion Waterfall">
        </div>
    </div>

    <!-- Drift Heatmap and KS Distributions -->
    <div class="grid-2">
        <div class="card">
            <div class="card-header">
                <h2>3. Multi-Month Covariate Drift Heatmap (vs. Jan 2026 Baseline)</h2>
            </div>
            <img class="responsive-fig" src="{img_heatmap}" alt="Sensor Drift Heatmap">
        </div>
        <div class="card">
            <div class="card-header">
                <h2>4. Baseline vs. Operational Shift Density Curves</h2>
            </div>
            <img class="responsive-fig" src="{img_distributions}" alt="KS Drift Distributions">
        </div>
    </div>

    <!-- Incident Log Table -->
    <div class="card">
        <div class="card-header">
            <h2>5. Recent DCS Operational Excursions & Local Root-Cause Log</h2>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Unit</th>
                    <th>Actual SEC</th>
                    <th>Z-Score</th>
                    <th>Severity</th>
                    <th>Failure Mode Classification</th>
                    <th>Top SHAP Drivers</th>
                    <th>Operator Guidance</th>
                </tr>
            </thead>
            <tbody>
                {incident_rows}
            </tbody>
        </table>
    </div>

    <!-- Feature Drift Diagnostics Table -->
    <div class="card">
        <div class="card-header">
            <h2>6. Kolmogorov-Smirnov Hypothesis Test Diagnostics Table (α = 0.05)</h2>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Monitored DCS Tag</th>
                    <th>KS Statistic (D)</th>
                    <th>p-Value</th>
                    <th>Baseline Mean</th>
                    <th>Monitoring Mean</th>
                    <th>Mean Shift (%)</th>
                    <th>Statistical Status</th>
                </tr>
            </thead>
            <tbody>
                {drift_rows}
            </tbody>
        </table>
    </div>

</div>

<footer>
    <p>Chambal Fertilisers and Chemicals Limited (CFCL) — Gadepan Complex (Kota, Rajasthan)<br>
    Specific Energy Consumption (SEC) Optimization Engine &copy; 2026. Confidential — Internal DCS Technical Report.</p>
</footer>

</body>
</html>
"""
        out_path = self.reports_dir / "drift_monitoring_dashboard.html"
        out_path.write_text(html_content, encoding="utf-8")
        self.logger.info("Saved interactive monitoring dashboard to %s (%d KB)", out_path, len(html_content) // 1024)
        return out_path


def main() -> None:
    """Generate executive monitoring dashboard CLI."""
    generator = MonitoringDashboardGenerator()
    generator.generate_html()


if __name__ == "__main__":
    main()
