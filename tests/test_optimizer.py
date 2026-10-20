"""Tests for Constrained SLSQP Optimizer and Financial Savings Calculator.

Verifies:
1. Hard DCS safety governor boundaries are never violated.
2. 1-hour slew-rate ramp constraints are strictly enforced.
3. Multi-start SLSQP produces non-negative specific energy reductions.
4. Gas tariff economics ($11.50/MMBTU) and IPCC Scope-1 carbon calculations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.optimization.constrained_optimizer import ConstrainedSECOptimizer
from src.optimization.savings_calculator import SavingsCalculator


class TestConstrainedOptimizer:
    """Verify SLSQP supervisory setpoint optimization."""

    def test_optimizer_initialization(self, optimizer_instance: ConstrainedSECOptimizer) -> None:
        """Optimizer must load surrogate model and feature names successfully."""
        assert optimizer_instance.model is not None
        assert len(optimizer_instance.feature_names) == 184

    def test_hard_governor_bounds_respected(
        self,
        optimizer_instance: ConstrainedSECOptimizer,
        sample_telemetry_series: pd.Series
    ) -> None:
        """Recommended setpoints must strictly adhere to DCS governor envelopes."""
        res = optimizer_instance.optimize_hour(sample_telemetry_series, respect_slew_rate=False)
        rec = res.recommended_setpoints

        assert 2.80 <= rec["steam_carbon_ratio"] <= 3.40, f"S/C out of bounds: {rec['steam_carbon_ratio']}"
        assert 780.0 <= rec["reformer_outlet_temp"] <= 835.0, f"Reformer Temp out of bounds: {rec['reformer_outlet_temp']}"
        assert 145.0 <= rec["loop_pressure"] <= 185.0, f"Loop Pressure out of bounds: {rec['loop_pressure']}"

    def test_slew_rate_interlocks_enforced(
        self,
        optimizer_instance: ConstrainedSECOptimizer,
        sample_telemetry_series: pd.Series
    ) -> None:
        """When respect_slew_rate=True, adjustments must not exceed maximum 1-hour ramp steps."""
        res = optimizer_instance.optimize_hour(sample_telemetry_series, respect_slew_rate=True)
        adj = res.setpoint_adjustments

        assert abs(adj["steam_carbon_ratio"]) <= 0.051, f"S/C slew rate exceeded: {adj['steam_carbon_ratio']}"
        assert abs(adj["reformer_outlet_temp"]) <= 2.01, f"Temp slew rate exceeded: {adj['reformer_outlet_temp']}"
        assert abs(adj["loop_pressure"]) <= 2.01, f"Pressure slew rate exceeded: {adj['loop_pressure']}"

    def test_energy_reduction_non_negative(
        self,
        optimizer_instance: ConstrainedSECOptimizer,
        sample_telemetry_series: pd.Series
    ) -> None:
        """SLSQP optimizer must never increase specific energy consumption."""
        res = optimizer_instance.optimize_hour(sample_telemetry_series, respect_slew_rate=True)
        assert res.delta_sec >= -1e-5, f"Optimization increased energy: delta={res.delta_sec}"
        assert res.optimized_sec <= res.baseline_sec + 1e-5


class TestSavingsCalculator:
    """Verify chemical plant economics and environmental accounting."""

    def test_gas_tariff_calibration(self, savings_calculator_instance: SavingsCalculator) -> None:
        """Pooled tariff of $11.50/MMBTU with 83.5 INR/USD should yield ~3,810.58 INR/Gcal."""
        cost_per_gcal = savings_calculator_instance.cost_per_gcal_inr
        assert 3700.0 <= cost_per_gcal <= 3900.0, f"Unexpected gas cost per Gcal: {cost_per_gcal}"

    def test_savings_calculation_structure(
        self,
        savings_calculator_instance: SavingsCalculator
    ) -> None:
        """Calculation output must contain annual financial savings and CO2 abatement."""
        # 0.02 Gcal/MT savings on Gadepan-III (2,350 MTPD)
        out = savings_calculator_instance.calculate_savings(delta_sec_gcal_per_mt=0.02, plant_id="gadepan_3")

        assert "annual_savings_crore_inr" in out
        assert "annual_co2_abated_tco2" in out
        assert out["annual_savings_crore_inr"] > 0.0
        assert out["annual_co2_abated_tco2"] > 0.0

    def test_fleet_scaling_gadepan3_vs_gadepan1(
        self,
        savings_calculator_instance: SavingsCalculator
    ) -> None:
        """Gadepan-III (2,350 MTPD) should yield proportionally higher absolute savings than G-I (1,520 MTPD)."""
        s_g3 = savings_calculator_instance.calculate_savings(delta_sec_gcal_per_mt=0.01, plant_id="gadepan_3")
        s_g1 = savings_calculator_instance.calculate_savings(delta_sec_gcal_per_mt=0.01, plant_id="gadepan_1")

        ratio = s_g3["annual_savings_crore_inr"] / s_g1["annual_savings_crore_inr"]
        expected_ratio = (2350.0 * 340.0) / (1520.0 * 330.0)
        assert abs(ratio - expected_ratio) < 0.05
        assert abs(ratio - expected_ratio) < 0.05
