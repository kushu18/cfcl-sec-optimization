"""
CFCL Gadepan Complex - Ammonia Synthesis Loop & Urea Stoichiometric Coupling Engine.
Temkin-Pyzhev loop kinetics, argon/methane inerts accumulation balance,
and strict CO2/NH3 stoichiometric tie-in for downstream urea synthesis.
Sanitized DCS Telemetry under strict NDA.
"""

import math
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class LoopMaterialBalanceResult:
    """Ammonia synthesis loop inerts accumulation and purge gas rate balance."""
    loop_pressure_bar: float
    loop_inerts_fraction_mol_pct: float
    ammonia_yield_mtpd: float
    purge_gas_rate_nm3_per_hr: float
    hydrogen_recovery_rate_nm3_per_hr: float
    syngas_compressor_power_mw: float
    is_loop_balanced: bool
    synthesis_efficiency_rating: str


@dataclass
class UreaStoichiometricResult:
    """CO2 / NH3 stoichiometric tie-in for downstream urea production."""
    ammonia_production_mtpd: float
    urea_target_mtpd: float
    co2_recovered_mtpd: float
    stoichiometric_co2_required_mtpd: float
    stoichiometric_nh3_required_mtpd: float
    co2_surplus_deficit_mtpd: float
    nh3_surplus_deficit_mtpd: float
    is_urea_stoichiometrically_satisfied: bool
    limiting_reagent: str
    max_achievable_urea_mtpd: float


class AmmoniaSynthesisLoopEngine:
    """
    Evaluates the high-pressure synthesis loop material balance and stoichiometric
    coupling to the urea melt and prilling plants across Gadepan-I, II, and III.

    Reactions:
      N2 + 3 H2 <-> 2 NH3  (Exothermic, Delta H = -92.4 kJ/mol)
      2 NH3 + CO2 <-> NH2COONH4 <-> NH2CONH2 + H2O (Urea Synthesis)

    Stoichiometric consumption:
      0.567 MT NH3 per MT Urea
      0.733 MT CO2 per MT Urea (Practical industrial norm: 0.735 - 0.745 MT/MT)
    """

    STOICH_NH3_PER_MT_UREA = 0.5671
    STOICH_CO2_PER_MT_UREA = 0.7350  # Includes ~0.5% stripping and vent losses

    def __init__(self, design_ammonia_capacity_mtpd: float = 1750.0):
        self.design_ammonia_capacity = design_ammonia_capacity_mtpd

    def evaluate_loop_balance(
        self,
        loop_pressure_bar: float,
        purge_gas_fraction: float = 0.045,
        catalyst_activity_factor: float = 0.95,
    ) -> LoopMaterialBalanceResult:
        """
        Calculates inerts accumulation (Ar + unreacted CH4) and loop yield.
        High loop pressure increases ammonia equilibrium conversion (Le Chatelier),
        but increases synthesis gas compressor energy consumption.
        Higher purge cuts loop inerts, increasing NH3 partial pressure, but vents syngas.
        """
        # Clamped operating range: 170 to 220 bar
        p_clamped = max(160.0, min(230.0, loop_pressure_bar))

        # Inerts fraction in circulating loop gas:
        # Inerts_in / Purge_fraction
        # Feed inerts typically ~ 0.35% Ar + 0.15% CH4 = 0.50%
        feed_inerts = 0.0050
        purge_frac_clamped = max(0.02, min(0.10, purge_gas_fraction))
        # Loop material balance: Inerts accumulate until Inerts_purge = Inerts_feed
        # y_loop_inerts = (feed_inerts / purge_frac) * (1 - single_pass_conversion)
        loop_inerts_mol_pct = (feed_inerts / purge_frac_clamped) * 0.835 * 100.0

        # Temkin-Pyzhev conversion approximation:
        # Higher pressure -> higher yield; Higher inerts -> lower yield
        base_conversion = 0.165  # 16.5% single-pass conversion
        press_factor = 1.0 + 0.0028 * (p_clamped - 190.0)
        inerts_penalty = 1.0 - 0.015 * (loop_inerts_mol_pct - 10.0)
        catalyst_factor = catalyst_activity_factor

        single_pass_conversion = base_conversion * press_factor * inerts_penalty * catalyst_factor
        single_pass_conversion = max(0.10, min(0.24, single_pass_conversion))

        # Ammonia production rate
        ammonia_yield = self.design_ammonia_capacity * (single_pass_conversion / base_conversion)

        # Compressor power estimation: ~ 420 - 480 kWh / MT NH3
        specific_power_kwh_mt = 410.0 + 0.55 * (p_clamped - 170.0)
        compressor_power_mw = (ammonia_yield * specific_power_kwh_mt) / (24.0 * 1000.0)

        # Purge gas flow rate (Nm3/hr)
        purge_gas_nm3_hr = (ammonia_yield * 1000.0 / 24.0) * 0.18 * (purge_frac_clamped / 0.045)
        # Hydrogen recovery unit (HRU) recovers ~88% of H2 from purge gas
        h2_recovered_nm3_hr = purge_gas_nm3_hr * 0.62 * 0.88

        is_balanced = 8.0 <= loop_inerts_mol_pct <= 14.0

        if loop_inerts_mol_pct < 8.0:
            rating = "SUB_OPTIMAL_HIGH_PURGE_SYNGAS_VENT"
        elif loop_inerts_mol_pct > 14.0:
            rating = "SUB_OPTIMAL_HIGH_INERTS_PRESSURE_CHOKING"
        else:
            rating = "OPTIMAL_LOOP_EQUILIBRIUM"

        return LoopMaterialBalanceResult(
            loop_pressure_bar=round(p_clamped, 2),
            loop_inerts_fraction_mol_pct=round(loop_inerts_mol_pct, 2),
            ammonia_yield_mtpd=round(ammonia_yield, 2),
            purge_gas_rate_nm3_per_hr=round(purge_gas_nm3_hr, 1),
            hydrogen_recovery_rate_nm3_per_hr=round(h2_recovered_nm3_hr, 1),
            syngas_compressor_power_mw=round(compressor_power_mw, 2),
            is_loop_balanced=is_balanced,
            synthesis_efficiency_rating=rating,
        )

    def evaluate_urea_stoichiometry(
        self,
        ammonia_production_mtpd: float,
        co2_recovered_mtpd: float,
        target_urea_mtpd: float,
    ) -> UreaStoichiometricResult:
        """
        Validates stoichiometric coupling between upstream ammonia synthesis
        and downstream urea manufacturing.

        Ensures optimizer setpoints never starve the urea plant of CO2 or NH3.
        """
        required_co2 = target_urea_mtpd * self.STOICH_CO2_PER_MT_UREA
        required_nh3 = target_urea_mtpd * self.STOICH_NH3_PER_MT_UREA

        co2_delta = co2_recovered_mtpd - required_co2
        nh3_delta = ammonia_production_mtpd - required_nh3

        max_urea_from_co2 = co2_recovered_mtpd / self.STOICH_CO2_PER_MT_UREA
        max_urea_from_nh3 = ammonia_production_mtpd / self.STOICH_NH3_PER_MT_UREA

        max_achievable_urea = min(max_urea_from_co2, max_urea_from_nh3)

        if max_urea_from_co2 < max_urea_from_nh3:
            limiting = "CO2_CARBON_LIMITED"
        elif max_urea_from_nh3 < max_urea_from_co2:
            limiting = "NH3_AMMONIA_LIMITED"
        else:
            limiting = "STRICTLY_BALANCED"

        is_satisfied = (co2_delta >= -10.0) and (nh3_delta >= -5.0)

        return UreaStoichiometricResult(
            ammonia_production_mtpd=round(ammonia_production_mtpd, 2),
            urea_target_mtpd=round(target_urea_mtpd, 2),
            co2_recovered_mtpd=round(co2_recovered_mtpd, 2),
            stoichiometric_co2_required_mtpd=round(required_co2, 2),
            stoichiometric_nh3_required_mtpd=round(required_nh3, 2),
            co2_surplus_deficit_mtpd=round(co2_delta, 2),
            nh3_surplus_deficit_mtpd=round(nh3_delta, 2),
            is_urea_stoichiometrically_satisfied=is_satisfied,
            limiting_reagent=limiting,
            max_achievable_urea_mtpd=round(max_achievable_urea, 2),
        )
