"""Industrial Financial & Carbon Abatement Savings Calculator for CFCL Gadepan.

Translates Specific Energy Consumption reductions (ΔSEC in Gcal/MT urea) into:
1. Natural Gas Fuel Savings:
   - Specific gas consumption savings (SCM / MT urea)
   - Annual volumetric natural gas savings (Million SCM / year)
   - Annual energy savings (Million MMBTU / year)
2. Tangible Financial Impact:
   - Cost savings per metric tonne of urea (₹ / MT urea)
   - Annualized fleet financial savings in ₹ Crore / year (1 Crore = ₹ 10 Million)
   - Calibrated against Indian fertilizer pooled gas price ($11.50/MMBTU @ ₹83.50/$)
3. Decarbonization & Environmental Sustainability (BRSR Aligned):
   - Scope 1 direct carbon emission abatement (tCO₂ / year)
   - Equivalent passenger vehicles removed from Rajasthan highways
   - Equivalent urban tree seedlings sequestering carbon over 10 years
4. Exports publication-quality CFCL-branded figures and structured JSON reports
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

from src.optimization.constrained_optimizer import ConstrainedSECOptimizer
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


class SavingsCalculator:
    """Enterprise financial and carbon abatement calculator calibrated for CFCL Gadepan."""

    # -------------------------------------------------------------------------
    # Physical, Economic & Environmental Calibration Constants
    # -------------------------------------------------------------------------
    # Gas Pricing: Indian Fertilizer Pooled Tariff (APM + RLNG blend)
    GAS_PRICE_USD_PER_MMBTU: float = 11.50      # ~$11.50 / MMBTU
    EXCHANGE_RATE_INR_PER_USD: float = 83.50     # ₹83.50 / USD
    MMBTU_PER_GCAL: float = 3.96832            # 1 Gcal = 3.96832 MMBTU
    
    # Gas Heating Value & Density
    GAS_LHV_KCAL_PER_SCM: float = 8600.0        # Average LHV of HBJ pipeline gas
    SCM_PER_GCAL: float = 1000000.0 / 8600.0    # ~116.28 SCM / Gcal

    # Plant Nameplate Capacities & Annual Production (CFCL Disclosures)
    PLANT_CAPACITIES_MTPD: Dict[str, float] = {
        "gadepan_1": 1520.0,    # 1,520 MTPD (1993 Snamprogetti)
        "gadepan_2": 1520.0,    # 1,520 MTPD (1999 Snamprogetti Revamp)
        "gadepan_3": 2350.0,    # 2,350 MTPD (2019 KBR Purifier / Toyo ACES 21)
    }

    STREAM_DAYS_PER_YEAR: Dict[str, float] = {
        "gadepan_1": 330.0,     # ~25-day annual turnaround allowance
        "gadepan_2": 330.0,     # ~25-day annual turnaround allowance
        "gadepan_3": 340.0,     # ~20-day annual turnaround allowance
    }

    # Carbon Abatement Factors (IPCC & GHG Protocol Scope 1 for Natural Gas)
    TCO2_PER_GCAL_GAS: float = 0.0558           # 0.0558 metric tonnes CO₂ per Gcal natural gas burned
    CAR_EMISSIONS_TCO2_YEAR: float = 4.6        # Standard Indian passenger vehicle annual emissions
    TREE_SEQUESTRATION_TCO2_YEAR: float = 0.022 # Urban tree annual CO₂ sequestration

    # Department of Fertilizers (DoF) New Urea Policy (NUP-2015) Target Energy Norms (Gcal/MT)
    NUP2015_ENERGY_NORMS: Dict[str, float] = {
        "gadepan_1": 5.500,
        "gadepan_2": 5.450,
        "gadepan_3": 5.000,
    }
    ESCERT_VALUE_INR: float = 2200.0  # BEE PAT Scheme market price per ESCert (10 Gcal = 1 MTOE)

    def __init__(self, logger: Optional[Any] = None) -> None:
        self.logger = logger or get_logger("savings_calculator")
        self.paths = get_paths()
        self.figures_dir = self.paths["figures"]
        self.reports_dir = self.paths["reports"]

        # Cost of energy in INR per Gcal
        self.cost_per_mmbtu_inr = self.GAS_PRICE_USD_PER_MMBTU * self.EXCHANGE_RATE_INR_PER_USD
        self.cost_per_gcal_inr = self.cost_per_mmbtu_inr * self.MMBTU_PER_GCAL
        self.cost_per_scm_inr = self.cost_per_gcal_inr / self.SCM_PER_GCAL

        self.logger.info("Calibrated Gas Tariff: $%.2f/MMBTU = ₹%.2f/Gcal (₹%.2f/SCM)",
                         self.GAS_PRICE_USD_PER_MMBTU, self.cost_per_gcal_inr, self.cost_per_scm_inr)

    def get_annual_production_mt(self, plant_id: Optional[str] = None) -> float:
        """Calculate design annual urea production in metric tonnes."""
        if plant_id is not None:
            p = plant_id.lower()
            return self.PLANT_CAPACITIES_MTPD.get(p, 1520.0) * self.STREAM_DAYS_PER_YEAR.get(p, 330.0)
        
        # Fleet-wide total
        return sum(
            self.PLANT_CAPACITIES_MTPD[p] * self.STREAM_DAYS_PER_YEAR[p]
            for p in self.PLANT_CAPACITIES_MTPD
        )

    def calculate_savings(self, delta_sec_gcal_per_mt: float, plant_id: Optional[str] = None) -> Dict[str, Any]:
        """Convert a given delta SEC reduction into financial and environmental savings."""
        annual_prod_mt = self.get_annual_production_mt(plant_id)

        # 1. Energy Savings
        annual_energy_saved_gcal = delta_sec_gcal_per_mt * annual_prod_mt
        annual_energy_saved_mmbtu = annual_energy_saved_gcal * self.MMBTU_PER_GCAL
        annual_gas_saved_scm = annual_energy_saved_gcal * self.SCM_PER_GCAL
        million_scm_saved = annual_gas_saved_scm / 1e6

        # 2. Financial Savings
        savings_per_mt_inr = delta_sec_gcal_per_mt * self.cost_per_gcal_inr
        annual_savings_inr = annual_energy_saved_gcal * self.cost_per_gcal_inr
        annual_savings_crore = annual_savings_inr / 1e7  # 1 Crore = 10,000,000 INR

        # 3. Decarbonization Impact
        annual_co2_abated_tonnes = annual_energy_saved_gcal * self.TCO2_PER_GCAL_GAS
        cars_removed = int(annual_co2_abated_tonnes / self.CAR_EMISSIONS_TCO2_YEAR)
        trees_equivalent = int(annual_co2_abated_tonnes / self.TREE_SEQUESTRATION_TCO2_YEAR)

        # 4. Sovereign Subsidy & BEE Energy Policy Accounting (NUP-2015 Tiered Clawback)
        nup_metrics = self.calculate_nup_retention(
            plant_id=plant_id or "gadepan_1",
            current_sec=5.500,  # Baseline norm reference
            delta_sec=delta_sec_gcal_per_mt,
            gross_savings_inr=annual_savings_inr,
        )
        escert_metrics = self.calculate_bee_escerts(annual_energy_saved_gcal)

        return {
            "plant_id": plant_id or "fleet_wide",
            "annual_production_mt": round(annual_prod_mt, 0),
            "delta_sec_gcal_per_mt": round(delta_sec_gcal_per_mt, 5),
            "savings_per_mt_inr": round(savings_per_mt_inr, 2),
            "annual_savings_crore_inr": round(annual_savings_crore, 3),
            "annual_savings_inr": round(annual_savings_inr, 0),
            "nup2015_retained_ebitda_crore": nup_metrics["retained_ebitda_crore"],
            "nup2015_sovereign_clawback_crore": nup_metrics["sovereign_clawback_crore"],
            "nup2015_retention_tier": nup_metrics["retention_tier"],
            "bee_pat_escerts_awarded": escert_metrics["escerts_count"],
            "bee_pat_revenue_crore": escert_metrics["escerts_revenue_crore"],
            "annual_gas_saved_million_scm": round(million_scm_saved, 3),
            "annual_energy_saved_mmbtu": round(annual_energy_saved_mmbtu, 1),
            "annual_co2_abated_tco2": round(annual_co2_abated_tonnes, 1),
            "equivalent_passenger_vehicles": cars_removed,
            "equivalent_trees_planted": trees_equivalent,
        }

    def calculate_nup_retention(
        self,
        plant_id: str,
        current_sec: float,
        delta_sec: float,
        gross_savings_inr: float,
    ) -> Dict[str, Any]:
        """
        Calculates corporate EBITDA retention under Department of Fertilizers NUP-2015.
        Tier 1: Actual > Norm: 100% retained (escapes subsidy penalty).
        Tier 2: Norm - 0.20 <= Actual <= Norm: 65% retained, 35% clawed back.
        Tier 3: Actual < Norm - 0.20: 33% retained, 67% passed through to Govt.
        """
        norm = self.NUP2015_ENERGY_NORMS.get(plant_id.lower(), 5.500)
        optimized_sec = current_sec - delta_sec

        if optimized_sec >= norm:
            retention_frac = 1.00
            tier = "TIER_0_PENALTY_AVOIDANCE_100PCT_RETAINED"
        elif optimized_sec >= (norm - 0.20):
            retention_frac = 0.65
            tier = "TIER_1_STANDARD_EFFICIENCY_65PCT_RETAINED"
        else:
            retention_frac = 0.33
            tier = "TIER_2_SUPER_EFFICIENCY_33PCT_RETAINED"

        retained_inr = gross_savings_inr * retention_frac
        clawback_inr = gross_savings_inr * (1.0 - retention_frac)

        return {
            "norm_gcal_mt": norm,
            "optimized_sec": round(optimized_sec, 4),
            "retention_fraction": retention_frac,
            "retention_tier": tier,
            "retained_ebitda_crore": round(retained_inr / 1e7, 3),
            "sovereign_clawback_crore": round(clawback_inr / 1e7, 3),
        }

    def calculate_bee_escerts(self, annual_energy_saved_gcal: float) -> Dict[str, Any]:
        """
        Calculates Energy Saving Certificates (ESCerts) under BEE Perform, Achieve and Trade (PAT).
        1 ESCert = 1 MTOE = 10 Gcal. Tradeable at IEX.
        """
        escerts_count = int(annual_energy_saved_gcal / 10.0)
        revenue_inr = escerts_count * self.ESCERT_VALUE_INR
        return {
            "escerts_count": escerts_count,
            "escerts_revenue_crore": round(revenue_inr / 1e7, 3),
            "escerts_revenue_inr": round(revenue_inr, 0),
        }

    def run_full_savings_assessment(self, batch_opt_df: pd.DataFrame) -> Dict[str, Any]:
        """Execute complete fleet financial and carbon audit from batch optimization results."""
        self.logger.info("Executing full savings and decarbonization assessment across %d optimized records...", len(batch_opt_df))

        # Mean delta SEC overall and per-plant
        fleet_mean_delta = float(batch_opt_df["delta_sec"].mean())

        # Fleet-level metrics
        fleet_savings = self.calculate_savings(fleet_mean_delta, plant_id=None)

        # Per-plant breakdown
        plant_savings: Dict[str, Dict[str, Any]] = {}
        for p_id in ["gadepan_1", "gadepan_2", "gadepan_3"]:
            p_sub = batch_opt_df[batch_opt_df["plant_id"] == p_id]
            p_delta = float(p_sub["delta_sec"].mean()) if len(p_sub) > 0 else fleet_mean_delta
            plant_savings[p_id] = self.calculate_savings(p_delta, plant_id=p_id)

        # Re-sum plant-specific totals for accurate fleet summary
        total_crore = sum(plant_savings[p]["annual_savings_crore_inr"] for p in plant_savings)
        total_co2 = sum(plant_savings[p]["annual_co2_abated_tco2"] for p in plant_savings)
        total_gas = sum(plant_savings[p]["annual_gas_saved_million_scm"] for p in plant_savings)

        report = {
            "assessment_timestamp": pd.Timestamp.now().isoformat(),
            "optimized_telemetry_hours": len(batch_opt_df),
            "gas_tariff": {
                "usd_per_mmbtu": self.GAS_PRICE_USD_PER_MMBTU,
                "exchange_rate_inr_usd": self.EXCHANGE_RATE_INR_PER_USD,
                "inr_per_gcal": round(self.cost_per_gcal_inr, 2),
                "inr_per_scm": round(self.cost_per_scm_inr, 2),
            },
            "fleet_summary": {
                "combined_capacity_mtpd": sum(self.PLANT_CAPACITIES_MTPD.values()),
                "total_annual_production_mt": sum(self.STREAM_DAYS_PER_YEAR[p] * self.PLANT_CAPACITIES_MTPD[p] for p in self.PLANT_CAPACITIES_MTPD),
                "mean_sec_reduction_gcal_per_mt": round(fleet_mean_delta, 5),
                "total_annual_savings_crore_inr": round(total_crore, 3),
                "total_annual_gas_saved_million_scm": round(total_gas, 3),
                "total_annual_co2_abated_tco2": round(total_co2, 1),
                "total_equivalent_passenger_vehicles": int(total_co2 / self.CAR_EMISSIONS_TCO2_YEAR),
                "total_equivalent_trees_planted": int(total_co2 / self.TREE_SEQUESTRATION_TCO2_YEAR),
            },
            "plants": plant_savings,
        }

        # Save JSON Report
        out_json = self.reports_dir / "optimization_savings_report.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        self.logger.info("Saved complete savings report to %s", out_json)

        # Generate plots
        self.plot_sec_reduction_distribution(batch_opt_df)
        self.plot_financial_savings_waterfall(plant_savings, total_crore)
        self.plot_setpoint_trajectory(batch_opt_df)
        self.plot_carbon_abatement_frontier(fleet_mean_delta)

        return report

    # =========================================================================
    # Visualizations (CFCL Branded Theme)
    # =========================================================================

    def plot_sec_reduction_distribution(self, opt_df: pd.DataFrame) -> Path:
        """Create publication-quality distribution plot of achievable SEC reductions."""
        apply_cfcl_theme()
        fig, ax = plt.subplots(figsize=(11, 6))

        deltas = opt_df["delta_sec"].values
        mean_d = float(np.mean(deltas))
        median_d = float(np.median(deltas))

        # Histogram
        n, bins, patches = ax.hist(deltas, bins=35, density=True, color=CFCL_LIGHT_BLUE,
                                   edgecolor=CFCL_DARK, alpha=0.7, label="Empirical SEC Delta (DCS Telemetry)")

        # KDE Curve
        kde = stats.gaussian_kde(deltas)
        x_grid = np.linspace(min(deltas), max(deltas), 150)
        ax.plot(x_grid, kde(x_grid), color=CFCL_BLUE, linewidth=2.5, label="Kernel Density Estimate (KDE)")

        # Reference lines
        ax.axvline(mean_d, color="#E74C3C", linestyle="--", linewidth=1.8, label=f"Mean Reduction: {mean_d:.4f} Gcal/MT")
        ax.axvline(median_d, color="#27AE60", linestyle=":", linewidth=1.8, label=f"Median Reduction: {median_d:.4f} Gcal/MT")

        ax.set_xlabel(r"Specific Energy Consumption Reduction ($\Delta$SEC, Gcal/MT urea)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
        ax.set_title("Distribution of Achievable SEC Reductions via Constrained SLSQP Optimizer",
                     fontsize=13, fontweight="bold", color=CFCL_BLUE, pad=12)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", framealpha=0.9, fontsize=10)
        plt.tight_layout()

        out_path = self.figures_dir / "optimizer_sec_reduction_distribution.png"
        save_figure(fig, out_path)
        plt.close(fig)
        self.logger.info("Saved SEC reduction distribution plot to %s", out_path)
        return out_path

    def plot_financial_savings_waterfall(self, plant_savings: Dict[str, Dict[str, Any]], total_crore: float) -> Path:
        """Create monetary savings waterfall chart across operational units."""
        apply_cfcl_theme()
        fig, ax = plt.subplots(figsize=(11, 6))

        labels = ["Gadepan-I\n(1,520 MTPD)", "Gadepan-II\n(1,520 MTPD)", "Gadepan-III\n(2,350 MTPD)", "CFCL Total Fleet\n(5,390 MTPD)"]
        values = [
            plant_savings["gadepan_1"]["annual_savings_crore_inr"],
            plant_savings["gadepan_2"]["annual_savings_crore_inr"],
            plant_savings["gadepan_3"]["annual_savings_crore_inr"],
            total_crore,
        ]
        colors = [PLANT_COLORS["gadepan_1"], PLANT_COLORS["gadepan_2"], PLANT_COLORS["gadepan_3"], "#27AE60"]

        bars = ax.bar(labels, values, color=colors, edgecolor=CFCL_DARK, width=0.55, linewidth=1.2)

        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + (total_crore * 0.02),
                    f"₹{h:.2f} Cr/yr", ha="center", va="bottom", fontsize=11, fontweight="bold", color=CFCL_DARK)

        ax.set_ylabel("Annualized Fuel Gas Cost Savings (₹ Crore / year)", fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(values) * 1.22)
        ax.set_title("CFCL Gadepan: Projected Annual Fuel Gas Savings by Operating Unit (@ $11.50/MMBTU)",
                     fontsize=13, fontweight="bold", color=CFCL_BLUE, pad=12)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        plt.tight_layout()

        out_path = self.figures_dir / "financial_savings_waterfall.png"
        save_figure(fig, out_path)
        plt.close(fig)
        self.logger.info("Saved financial savings waterfall plot to %s", out_path)
        return out_path

    def plot_setpoint_trajectory(self, opt_df: pd.DataFrame) -> Path:
        """Plot a 7-day operational trajectory showing actual vs. recommended setpoints."""
        apply_cfcl_theme()
        # Take a continuous 7-day window (168 hours) for Gadepan-3 or first available plant
        p3_df = opt_df[opt_df["plant_id"] == "gadepan_3"].head(168)
        if len(p3_df) < 24:
            p3_df = opt_df.head(168)

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        time_steps = np.arange(len(p3_df))

        # 1. Steam-to-Carbon Ratio
        axes[0].plot(time_steps, p3_df["cur_sc"], color="#7F8C8D", linestyle="--", linewidth=1.5, label="Measured DCS Setpoint")
        axes[0].plot(time_steps, p3_df["opt_sc"], color=CFCL_BLUE, linewidth=2.0, label="Recommended Optimal Setpoint")
        axes[0].set_ylabel("S/C Ratio (mol/mol)", fontsize=10, fontweight="bold")
        axes[0].set_title("Steam-to-Carbon Ratio Trajectory (Slew Rate: ≤0.05/hr)", fontsize=11, fontweight="bold")
        axes[0].grid(True, linestyle=":", alpha=0.6)
        axes[0].legend(loc="upper right", fontsize=9)

        # 2. Reformer Outlet Temp
        axes[1].plot(time_steps, p3_df["cur_reformer_temp"], color="#7F8C8D", linestyle="--", linewidth=1.5, label="Measured DCS Setpoint")
        axes[1].plot(time_steps, p3_df["opt_reformer_temp"], color="#E67E22", linewidth=2.0, label="Recommended Optimal Setpoint")
        axes[1].set_ylabel("Reformer Temp (°C)", fontsize=10, fontweight="bold")
        axes[1].set_title("Primary Reformer Outlet Temperature Trajectory (Slew Rate: ≤2.0 °C/hr)", fontsize=11, fontweight="bold")
        axes[1].grid(True, linestyle=":", alpha=0.6)
        axes[1].legend(loc="upper right", fontsize=9)

        # 3. Synthesis Loop Pressure
        axes[2].plot(time_steps, p3_df["cur_loop_pressure"], color="#7F8C8D", linestyle="--", linewidth=1.5, label="Measured DCS Setpoint")
        axes[2].plot(time_steps, p3_df["opt_loop_pressure"], color="#27AE60", linewidth=2.0, label="Recommended Optimal Setpoint")
        axes[2].set_ylabel("Pressure (kg/cm²)", fontsize=10, fontweight="bold")
        axes[2].set_xlabel("Operational Time Horizon (Hours)", fontsize=11, fontweight="bold")
        axes[2].set_title("Synthesis Loop Pressure Trajectory (Slew Rate: ≤2.0 kg/cm²/hr)", fontsize=11, fontweight="bold")
        axes[2].grid(True, linestyle=":", alpha=0.6)
        axes[2].legend(loc="upper right", fontsize=9)

        plt.suptitle("DCS Closed-Loop Supervisory Recommendation Trajectory (Gadepan-III)",
                     fontsize=14, fontweight="bold", color=CFCL_BLUE, y=0.995)
        plt.tight_layout()

        out_path = self.figures_dir / "setpoint_recommendations_trajectory.png"
        save_figure(fig, out_path)
        plt.close(fig)
        self.logger.info("Saved setpoint trajectory plot to %s", out_path)
        return out_path

    def plot_carbon_abatement_frontier(self, mean_delta_sec: float) -> Path:
        """Create parametric carbon abatement and monetary savings frontier curve."""
        apply_cfcl_theme()
        fig, ax1 = plt.subplots(figsize=(11, 6))

        deltas = np.linspace(0.001, 0.050, 100)
        annual_prod = self.get_annual_production_mt(None)

        annual_crore = [self.calculate_savings(d, None)["annual_savings_crore_inr"] for d in deltas]
        annual_co2 = [self.calculate_savings(d, None)["annual_co2_abated_tco2"] for d in deltas]

        # Primary Axis: Financial Savings
        line1 = ax1.plot(deltas, annual_crore, color=CFCL_BLUE, linewidth=2.5, label="Financial Fuel Savings (₹ Crore/yr)")
        ax1.set_xlabel(r"Specific Energy Consumption Reduction ($\Delta$SEC, Gcal/MT urea)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Annual Monetary Savings (₹ Crore / year)", color=CFCL_BLUE, fontsize=11, fontweight="bold")
        ax1.tick_params(axis="y", labelcolor=CFCL_BLUE)
        ax1.grid(True, linestyle=":", alpha=0.6)

        # Secondary Axis: Carbon Abatement
        ax2 = ax1.twinx()
        line2 = ax2.plot(deltas, annual_co2, color="#27AE60", linewidth=2.5, linestyle="--", label="Scope 1 Carbon Abatement (tCO₂/yr)")
        ax2.set_ylabel("Annual Carbon Abatement (tCO₂ / year)", color="#27AE60", fontsize=11, fontweight="bold")
        ax2.tick_params(axis="y", labelcolor="#27AE60")

        # Current Operational Point
        curr_crore = self.calculate_savings(mean_delta_sec, None)["annual_savings_crore_inr"]
        curr_co2 = self.calculate_savings(mean_delta_sec, None)["annual_co2_abated_tco2"]
        ax1.plot(mean_delta_sec, curr_crore, "ro", markersize=8)
        ax1.annotate(f"Achieved: ₹{curr_crore:.2f} Cr/yr\n({curr_co2:,.0f} tCO₂/yr)",
                     xy=(mean_delta_sec, curr_crore),
                     xytext=(mean_delta_sec + 0.004, curr_crore * 0.8),
                     arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
                     fontsize=9, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.3", fc="#FCF3CF", ec="#B7950B"))

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper left", framealpha=0.9, fontsize=10)

        plt.title("CFCL Gadepan: Decarbonization & Financial Abatement Frontier Curve",
                  fontsize=13, fontweight="bold", color=CFCL_BLUE, pad=12)
        plt.tight_layout()

        out_path = self.figures_dir / "carbon_abatement_frontier.png"
        save_figure(fig, out_path)
        plt.close(fig)
        self.logger.info("Saved carbon abatement frontier plot to %s", out_path)
        return out_path


def main() -> None:
    """Run full optimization and savings pipeline."""
    paths = get_paths()
    df = pd.read_parquet(paths["processed_data"] / "plant_features.parquet")
    
    # Stratified sample across all 3 plants for fast and representative evaluation
    sample_df = df.groupby("plant_id").tail(30).reset_index(drop=True)
    
    optimizer = ConstrainedSECOptimizer()
    opt_results = optimizer.optimize_batch(sample_df)

    calculator = SavingsCalculator()
    report = calculator.run_full_savings_assessment(opt_results)
    
    print("\n" + "=" * 60)
    print("CFCL GADEPAN - ANNUAL OPTIMIZATION SAVINGS SUMMARY")
    print("=" * 60)
    fs = report["fleet_summary"]
    print(f"Total Fleet Production:    {fs['total_annual_production_mt']:,.0f} MT/year")
    print(f"Mean SEC Reduction:        {fs['mean_sec_reduction_gcal_per_mt']:.5f} Gcal/MT")
    print(f"Annual Financial Savings:  INR {fs['total_annual_savings_crore_inr']:.2f} Crore / year")
    print(f"Natural Gas Saved:         {fs['total_annual_gas_saved_million_scm']:.2f} Million SCM / year")
    print(f"Carbon Dioxide Abated:     {fs['total_annual_co2_abated_tco2']:,.1f} tCO₂ / year")
    print(f"Equivalent Cars Off Road:  {fs['total_equivalent_passenger_vehicles']:,} vehicles")
    print("=" * 60)


if __name__ == "__main__":
    main()
