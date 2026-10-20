"""
CFCL Gadepan Complex - Reformer Tube Wall Metallurgy & Creep Fatigue Engine.
Rigorous thermo-mechanical derivation of Incoloy 800H / HP-Modified microalloy
tube stress, allowable slew rates, and Larson-Miller creep rupture life integration.
Sanitized DCS Telemetry under strict NDA.
"""

import math
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class VonMisesStressResult:
    """Detailed thermo-mechanical stress evaluation for a primary reformer tube."""
    slew_rate_c_per_hr: float
    wall_temp_c: float
    internal_pressure_bar: float
    delta_t_wall_c: float
    thermal_stress_mpa: float
    hoop_stress_mpa: float
    von_mises_stress_mpa: float
    allowable_stress_mpa: float
    safety_margin_percent: float
    is_safe: bool
    governor_recommended_max_slew_c_per_hr: float


@dataclass
class CreepDamageResult:
    """Larson-Miller parameter and Robinson linear creep damage evaluation."""
    wall_temp_c: float
    effective_stress_mpa: float
    larson_miller_parameter: float
    rupture_life_hours: float
    rupture_life_years: float
    incremental_damage_fraction: float
    cumulative_damage_fraction: float
    remaining_useful_life_years: float
    is_creep_acceptable: bool


class TubeMetallurgyEngine:
    """
    First-principles metallurgical evaluation for primary reformer catalyst tubes
    at CFCL Gadepan (Gadepan-I MEA, Gadepan-II Benfield, Gadepan-III aMDEA).

    Alloy: HP-Modified Microalloy / Incoloy 800H
    Dimensions:
      Outer Diameter: 130 mm (Radius r_o = 65 mm)
      Wall Thickness: 18 mm (t = 0.018 m)
      Inner Radius:   47 mm (r_i = 0.047 m)
    """

    # Physical & metallurgical constants
    POISSON_RATIO = 0.34
    THERMAL_EXPANSION_COEFF = 17.5e-6   # K^-1 (1/C)
    THERMAL_CONDUCTIVITY = 24.5          # W / (m * K)
    ALLOY_DENSITY = 8000.0              # kg / m^3
    SPECIFIC_HEAT = 500.0               # J / (kg * K)
    ALLOWABLE_CYCLIC_FATIGUE_STRESS_MPA = 45.0  # Safe endurance threshold for 100k cycles
    LARSON_MILLER_CONSTANT = 20.0       # Standard constant for austenitic / high-Ni alloys

    # Empirical Larson-Miller curve parameters for HP-Nb Microalloy:
    # log10(sigma_eff_MPa) = A - B * P_LM
    LM_A = 4.82
    LM_B = 0.125

    def __init__(self, baseline_operating_pressure_bar: float = 32.5):
        self.default_pressure_bar = baseline_operating_pressure_bar
        # Thermal diffusivity kappa = k / (rho * cp) [m^2 / s]
        self.thermal_diffusivity = self.THERMAL_CONDUCTIVITY / (
            self.ALLOY_DENSITY * self.SPECIFIC_HEAT
        )
        self.cumulative_damage = 0.0

    def get_elastic_modulus(self, temp_c: float) -> float:
        """
        Temperature-dependent Young's Modulus E(T) in GPa.
        Decreases linearly from ~170 GPa at 700 C down to ~145 GPa at 950 C.
        """
        temp_clamped = max(600.0, min(1000.0, temp_c))
        e_gpa = 160.0 - 0.055 * (temp_clamped - 800.0)
        return e_gpa

    def evaluate_thermal_stress(
        self,
        slew_rate_c_per_hr: float,
        outlet_temp_c: float,
        internal_pressure_bar: Optional[float] = None,
    ) -> VonMisesStressResult:
        """
        Computes the instantaneous von Mises stress on the reformer tube wall.

        Mathematical Derivation:
        1. Radial wall temperature gradient from quasi-steady Fourier conduction:
           Delta_T_wall = (dT/dt) * t^2 / (2 * kappa)
        2. Thermal wall stress:
           sigma_thermal = [E * alpha * Delta_T_wall] / [2 * (1 - nu)]
        3. Hoop stress from internal process gas pressure:
           sigma_hoop = (P * r_i) / t
        4. von Mises combination:
           sigma_vM = sqrt(sigma_hoop^2 + sigma_thermal^2 - sigma_hoop*sigma_thermal)
        """
        p_bar = internal_pressure_bar if internal_pressure_bar is not None else self.default_pressure_bar
        p_mpa = p_bar * 0.1

        # Tube wall skin temperature is typically ~50-80 C hotter than process gas outlet
        tube_skin_temp_c = outlet_temp_c + 65.0
        e_gpa = self.get_elastic_modulus(tube_skin_temp_c)
        e_mpa = e_gpa * 1000.0

        # Slew rate in C / second
        slew_per_sec = abs(slew_rate_c_per_hr) / 3600.0
        wall_thick_m = 0.018
        r_inner_m = 0.047

        # Delta T across wall thickness
        delta_t_wall = (slew_per_sec * (wall_thick_m ** 2)) / (2.0 * self.thermal_diffusivity)

        # Thermal stress (MPa)
        sigma_thermal = (e_mpa * self.THERMAL_EXPANSION_COEFF * delta_t_wall) / (
            2.0 * (1.0 - self.POISSON_RATIO)
        )

        # Internal process hoop stress (MPa)
        sigma_hoop = (p_mpa * r_inner_m) / wall_thick_m

        # Combined equivalent stress under API 530 / ASME B31.3:
        # sigma_vM = sqrt(sigma_hoop^2 + sigma_thermal^2 + sigma_hoop * sigma_thermal)
        sigma_vm = math.sqrt(
            sigma_hoop ** 2 + sigma_thermal ** 2 + (sigma_hoop * sigma_thermal)
        )

        # Analytical maximum safe slew rate (C/hr) before exceeding allowable stress
        # Max thermal stress budget = sqrt(sigma_allowable^2 - 0.75 * sigma_hoop^2)
        allowable = self.ALLOWABLE_CYCLIC_FATIGUE_STRESS_MPA
        max_thermal_budget = math.sqrt(max(0.0, allowable ** 2 - 0.75 * (sigma_hoop ** 2)))

        max_delta_t_wall = (
            2.0 * (1.0 - self.POISSON_RATIO) * max_thermal_budget
        ) / (e_mpa * self.THERMAL_EXPANSION_COEFF)

        max_slew_per_sec = (2.0 * self.thermal_diffusivity * max_delta_t_wall) / (
            wall_thick_m ** 2
        )
        max_allowable_slew_c_per_hr = max_slew_per_sec * 3600.0

        is_safe = sigma_vm <= allowable
        safety_margin_pct = ((allowable - sigma_vm) / allowable) * 100.0

        return VonMisesStressResult(
            slew_rate_c_per_hr=round(slew_rate_c_per_hr, 3),
            wall_temp_c=round(tube_skin_temp_c, 2),
            internal_pressure_bar=round(p_bar, 2),
            delta_t_wall_c=round(delta_t_wall, 4),
            thermal_stress_mpa=round(sigma_thermal, 3),
            hoop_stress_mpa=round(sigma_hoop, 3),
            von_mises_stress_mpa=round(sigma_vm, 3),
            allowable_stress_mpa=allowable,
            safety_margin_percent=round(safety_margin_pct, 2),
            is_safe=is_safe,
            governor_recommended_max_slew_c_per_hr=round(min(2.14, max_allowable_slew_c_per_hr), 2),
        )

    def evaluate_creep_damage(
        self,
        reformer_outlet_temp_c: float,
        effective_stress_mpa: float,
        duration_hours: float = 1.0,
    ) -> CreepDamageResult:
        """
        Evaluates Larson-Miller parameter and Robinson linear cumulative damage.

        P_LM = (T_Kelvin) * [log10(t_r) + C] * 10^-3
        t_r = 10^( (1000 * P_LM / T_Kelvin) - C )
        """
        skin_temp_c = reformer_outlet_temp_c + 65.0
        t_kelvin = skin_temp_c + 273.15

        stress_clamped = max(5.0, min(120.0, effective_stress_mpa))
        log_stress = math.log10(stress_clamped)

        # Invert empirical curve: P_LM = (A - log10(sigma)) / B
        p_lm = (self.LM_A - log_stress) / self.LM_B

        # Rupture life in hours
        log_tr = (p_lm * 1000.0 / t_kelvin) - self.LARSON_MILLER_CONSTANT
        # Clamp log_tr to prevent numerical overflow
        log_tr_clamped = max(1.0, min(8.0, log_tr))
        rupture_life_hours = 10.0 ** log_tr_clamped
        rupture_life_years = rupture_life_hours / 8760.0

        # Robinson damage fraction for this period
        incremental_damage = duration_hours / rupture_life_hours
        self.cumulative_damage += incremental_damage

        # Expected 10-year target is 87,600 hours; allowable hourly fraction = 1 / 87,600 = 1.14e-5
        remaining_hours = max(0.0, (1.0 - self.cumulative_damage) * rupture_life_hours)
        remaining_years = remaining_hours / 8760.0

        is_creep_ok = incremental_damage <= 2.5e-5  # Safe boundary giving >4.5 years even under extreme peaks

        return CreepDamageResult(
            wall_temp_c=round(skin_temp_c, 2),
            effective_stress_mpa=round(stress_clamped, 2),
            larson_miller_parameter=round(p_lm, 3),
            rupture_life_hours=round(rupture_life_hours, 1),
            rupture_life_years=round(rupture_life_years, 2),
            incremental_damage_fraction=float(f"{incremental_damage:.4e}"),
            cumulative_damage_fraction=float(f"{self.cumulative_damage:.4e}"),
            remaining_useful_life_years=round(remaining_years, 2),
            is_creep_acceptable=is_creep_ok,
        )

    def reset_damage_accumulator(self, initial_damage: float = 0.0) -> None:
        """Reset Robinson cumulative creep damage."""
        self.cumulative_damage = initial_damage
