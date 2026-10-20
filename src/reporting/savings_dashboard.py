"""Interactive Executive Savings & Optimization Dashboard for CFCL Gadepan.

Generates a standalone, self-contained HTML executive report combining:
- Executive Financial KPI Summary Cards (₹ Crore/year, Million SCM gas saved)
- Decarbonization & ESG Impact Badges (Scope 1 tCO₂ abated, tree equivalents)
- Plant-by-Plant Economics Breakdown (Gadepan-I, II, and III)
- DCS Closed-Loop Trajectory & Safety Governor Interlocks Table
- Embedded high-resolution publication figures
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.config_loader import get_paths, get_project_root
from src.utils.logger import get_logger


class SavingsDashboardGenerator:
    """Generates an executive-ready HTML dashboard for Part 7 optimization & savings."""

    def __init__(self, logger: Optional[Any] = None) -> None:
        self.logger = logger or get_logger("savings_dashboard")
        self.paths = get_paths()
        self.figures_dir = self.paths["figures"]
        self.reports_dir = self.paths["reports"]

    def _encode_image(self, filename: str) -> str:
        """Encode a figure as a base64 data URI for self-contained HTML rendering."""
        img_path = self.figures_dir / filename
        if not img_path.exists():
            img_path = self.figures_dir / f"{filename}.png"
        if img_path.exists():
            data = img_path.read_bytes()
            b64 = base64.b64encode(data).decode("utf-8")
            return f"data:image/png;base64,{b64}"
        return ""

    def generate_html(self) -> Path:
        """Build and save the self-contained HTML executive savings report."""
        self.logger.info("Generating interactive executive optimization savings dashboard...")

        # Load Savings Report
        sav_path = self.reports_dir / "optimization_savings_report.json"
        with open(sav_path, "r", encoding="utf-8") as f:
            rep = json.load(f)

        fs = rep["fleet_summary"]
        tariff = rep["gas_tariff"]

        # Encode key figures
        img_waterfall = self._encode_image("financial_savings_waterfall.png")
        img_trajectory = self._encode_image("setpoint_recommendations_trajectory.png")
        img_distribution = self._encode_image("optimizer_sec_reduction_distribution.png")
        img_frontier = self._encode_image("carbon_abatement_frontier.png")

        # Build Plant Comparison Table Rows
        plant_rows = ""
        plant_display = {
            "gadepan_1": "Gadepan-I (1,520 MTPD, Topsoe / MEA, 1993)",
            "gadepan_2": "Gadepan-II (1,520 MTPD, Snamprogetti / Benfield, 1999)",
            "gadepan_3": "Gadepan-III (2,350 MTPD, KBR Purifier / aMDEA, 2019)",
        }

        for p_id, p_data in rep["plants"].items():
            name = plant_display.get(p_id, p_id)
            plant_rows += f"""
            <tr>
                <td><strong>{name}</strong></td>
                <td>{p_data['annual_production_mt']:,.0f} MT/yr</td>
                <td><code>{p_data['delta_sec_gcal_per_mt']:.5f} Gcal/MT</code></td>
                <td>₹{p_data['savings_per_mt_inr']:.2f} / MT</td>
                <td style="color: #27AE60; font-weight: bold;">₹{p_data['annual_savings_crore_inr']:.3f} Cr/yr</td>
                <td>{p_data['annual_gas_saved_million_scm']:.3f} M-SCM</td>
                <td style="font-weight: bold; color: #1E8449;">{p_data['annual_co2_abated_tco2']:,.1f} tCO₂/yr</td>
                <td>{p_data['equivalent_trees_planted']:,} trees</td>
            </tr>
            """

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CFCL Gadepan - Energy Optimization & Decarbonization Executive Report</title>
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
        .kpi-card.success {{
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
            <h1>CFCL Gadepan - Energy Optimization & Decarbonization Engine</h1>
            <p>Constrained SLSQP Optimization, Fuel Gas Savings & Scope 1 Decarbonization Audit</p>
        </div>
        <div class="header-badge">
            <strong>Facility:</strong> Gadepan (Kota, Rajasthan) &nbsp;|&nbsp; <strong>Fleet Capacity:</strong> 5,390 MTPD Urea
        </div>
    </div>
</header>

<div class="container">

    <!-- KPI Summary Cards -->
    <div class="kpi-grid">
        <div class="kpi-card success">
            <div class="kpi-label">Annual Financial Gas Savings</div>
            <div class="kpi-value">₹{fs['total_annual_savings_crore_inr']:.2f} Cr/yr</div>
            <div class="kpi-subtext">Immediate 1-hour ramp step savings (₹{fs['total_annual_savings_crore_inr']*100:.1f} Lakhs/yr)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Natural Gas Saved</div>
            <div class="kpi-value">{fs['total_annual_gas_saved_million_scm']:.3f} M-SCM</div>
            <div class="kpi-subtext">Million Standard Cubic Meters saved annually (@ {tariff['inr_per_scm']:.2f} ₹/SCM)</div>
        </div>
        <div class="kpi-card success">
            <div class="kpi-label">Scope 1 Carbon Abatement</div>
            <div class="kpi-value">{fs['total_annual_co2_abated_tco2']:,.1f} tCO₂/yr</div>
            <div class="kpi-subtext">Direct combustion emissions avoided ({fs['total_equivalent_trees_planted']:,} tree seedlings)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Gas Tariff Calibration</div>
            <div class="kpi-value">${tariff['usd_per_mmbtu']:.2f}</div>
            <div class="kpi-subtext">USD/MMBTU pooled fertilizer tariff (₹{tariff['inr_per_gcal']:.2f} / Gcal)</div>
        </div>
    </div>

    <!-- Plant-by-Plant Economics Table -->
    <div class="card">
        <div class="card-header">
            <h2>1. Plant-by-Plant Financial & Decarbonization Audit (Gadepan-I, II, and III)</h2>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Operating Unit</th>
                    <th>Annual Urea Capacity</th>
                    <th>Mean ΔSEC Reduction</th>
                    <th>Cost Saving / MT</th>
                    <th>Annual Monetary Saving</th>
                    <th>Annual Fuel Gas Saved</th>
                    <th>Carbon Abatement</th>
                    <th>Tree Equivalent</th>
                </tr>
            </thead>
            <tbody>
                {plant_rows}
            </tbody>
        </table>
    </div>

    <!-- Savings Waterfall & Carbon Frontier -->
    <div class="grid-2">
        <div class="card">
            <div class="card-header">
                <h2>2. Projected Annual Fuel Gas Savings by Operating Unit</h2>
            </div>
            <img class="responsive-fig" src="{img_waterfall}" alt="Financial Savings Waterfall">
        </div>
        <div class="card">
            <div class="card-header">
                <h2>3. Decarbonization & Abatement Frontier Curve</h2>
            </div>
            <img class="responsive-fig" src="{img_frontier}" alt="Carbon Abatement Frontier">
        </div>
    </div>

    <!-- Setpoint Trajectory and SEC Distribution -->
    <div class="grid-2">
        <div class="card">
            <div class="card-header">
                <h2>4. DCS Closed-Loop Trajectory (Gadepan-III, 7-Day Window)</h2>
            </div>
            <img class="responsive-fig" src="{img_trajectory}" alt="Setpoint Recommendations Trajectory">
        </div>
        <div class="card">
            <div class="card-header">
                <h2>5. Achievable Specific Energy Reductions (ΔSEC Distribution)</h2>
            </div>
            <img class="responsive-fig" src="{img_distribution}" alt="SEC Reduction Distribution">
        </div>
    </div>

    <!-- DCS Process Safety Interlocks Table -->
    <div class="card">
        <div class="card-header">
            <h2>6. DCS Safety Governor Interlocks & Slew-Rate Constraints</h2>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Process Variable</th>
                    <th>Safety Lower Bound</th>
                    <th>Safety Upper Bound</th>
                    <th>Maximum Hourly Slew Rate</th>
                    <th>Physical Engineering Justification</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Steam-to-Carbon Ratio</strong></td>
                    <td>2.80 mol/mol</td>
                    <td>3.40 mol/mol</td>
                    <td>±0.05 / hour</td>
                    <td>Below 2.80 risks catalytic carbon coking; above 3.40 overloads convection duty.</td>
                </tr>
                <tr>
                    <td><strong>Reformer Outlet Temperature</strong></td>
                    <td>780.0 °C</td>
                    <td>835.0 °C</td>
                    <td>±2.0 °C / hour</td>
                    <td>Prevents thermal stress & metallurgical creep fatigue on Incoloy 800H catalyst tubes.</td>
                </tr>
                <tr>
                    <td><strong>Synthesis Loop Pressure</strong></td>
                    <td>145.0 kg/cm²</td>
                    <td>185.0 kg/cm²</td>
                    <td>±2.0 kg/cm² / hour</td>
                    <td>Maintains syngas compressor surge margin and ammonia condensation equilibrium.</td>
                </tr>
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
        out_path = self.reports_dir / "optimization_savings_dashboard.html"
        out_path.write_text(html_content, encoding="utf-8")
        self.logger.info("Saved interactive savings dashboard to %s (%d KB)", out_path, len(html_content) // 1024)
        return out_path


def main() -> None:
    """Generate executive savings dashboard CLI."""
    generator = SavingsDashboardGenerator()
    generator.generate_html()


if __name__ == "__main__":
    main()
