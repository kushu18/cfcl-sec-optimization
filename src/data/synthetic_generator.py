"""Synthetic DCS data generator for the CFCL SEC Optimization pipeline.

Generates 6 months of hourly plant data for all 3 Gadepan units with
7 realistic temporal patterns baked into the signal:

    1. Diurnal cycle      — SEC rises during afternoon peak (via ambient/CW temp)
    2. Summer heat effect  — +0.06-0.10 Gcal/MT in Apr-May (extreme ambient temp)
    3. Catalyst decay     — Exponential SEC drift post-turnaround with TA reset
    4. Night shift drift  — ~2 % higher SEC during 22:00-06:00
    5. Weekend stability  — Tighter process variance on Sat/Sun
    6. Anomaly injection  — 3 compressor-trip SEC spikes per plant (2-4 hr)
    7. Technology gap     — G-III ~0.15-0.25 below G-I (aMDEA vs MEA benchmark)

Environmental signals (ambient temperature, humidity, natural-gas calorific
value) are generated **once** and shared across all 3 plants with per-plant
DCS measurement noise, reflecting the fact that all units sit in the same
Gadepan complex and draw gas from the same HBJ pipeline.

**Turnaround handling**: Each plant has a scheduled annual turnaround
(shutdown for catalyst replacement + maintenance). Data is not generated
during turnaround periods, and catalyst age resets to 0 afterwards.

Data is calibrated against publicly available CFCL Annual Report values
and standard Indian gas-based urea plant operating parameters.
Synthetic data is used to protect proprietary DCS readings per NDA.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from src.utils.config_loader import load_config, get_project_root
from src.utils.logger import get_logger

log = get_logger(__name__)


# =====================================================================
# Constants -- Climate, SEC Model, Plant Parameters
# =====================================================================

# Kota, Rajasthan monthly climate reference (IMD public data, all 12 months)
# temp_amp = diurnal half-amplitude, hum_amp = diurnal humidity swing
_KOTA_CLIMATE: dict[int, dict[str, float]] = {
    1:  {"temp_mean": 16.0, "temp_amp": 10.0, "hum_mean": 30.0, "hum_amp": 5.0},
    2:  {"temp_mean": 20.0, "temp_amp": 10.0, "hum_mean": 25.0, "hum_amp": 5.0},
    3:  {"temp_mean": 27.0, "temp_amp": 10.0, "hum_mean": 18.0, "hum_amp": 4.0},
    4:  {"temp_mean": 35.0, "temp_amp": 9.0,  "hum_mean": 14.0, "hum_amp": 3.0},
    5:  {"temp_mean": 40.0, "temp_amp": 8.0,  "hum_mean": 20.0, "hum_amp": 4.0},
    6:  {"temp_mean": 38.0, "temp_amp": 7.0,  "hum_mean": 42.0, "hum_amp": 8.0},
    7:  {"temp_mean": 32.0, "temp_amp": 6.5,  "hum_mean": 65.0, "hum_amp": 10.0},
    8:  {"temp_mean": 30.0, "temp_amp": 5.5,  "hum_mean": 72.0, "hum_amp": 8.0},
    9:  {"temp_mean": 30.0, "temp_amp": 7.0,  "hum_mean": 58.0, "hum_amp": 9.0},
    10: {"temp_mean": 29.0, "temp_amp": 9.0,  "hum_mean": 42.0, "hum_amp": 7.0},
    11: {"temp_mean": 24.0, "temp_amp": 10.0, "hum_mean": 32.0, "hum_amp": 6.0},
    12: {"temp_mean": 18.0, "temp_amp": 10.0, "hum_mean": 28.0, "hum_amp": 5.0},
}

# SEC model -- linear sensitivity per unit of *normalised* deviation.
# norm_dev = (actual - operating_mean) / (safety_max - safety_min)
# Positive coeff -> higher value increases SEC.
_SEC_LINEAR: dict[str, float] = {
    "steam_carbon_ratio":   +0.12,
    "loop_pressure":        +0.09,
    "reformer_outlet_temp": -0.12,
    "cw_inlet_temp":        +0.10,
    "ambient_temp":         +0.08,
    "ambient_humidity":     +0.06,
    "co2_slip_ppm":         +0.05,
    "ng_calorific_value":   -0.07,
    # catalyst_age_days handled separately (exponential decay model)
}

_NIGHT_SHIFT_PENALTY: float = 0.10
_NOISE_STD: float = 0.020
_WEEKEND_NOISE_FACTOR: float = 0.50

# Plant-specific DCS control tightness (multiplier on variable std)
_CONTROL_TIGHTNESS: dict[str, float] = {
    "gadepan_1": 1.10,  # 1993 vintage
    "gadepan_2": 1.00,  # 1999 baseline
    "gadepan_3": 0.75,  # 2019 modern DCS
}

# Catalyst age on Jan 1, 2026 (days since previous year's turnaround)
# G-I:   TA ended ~Apr 29, 2025  -> Jan 1 2026 = 247 days
# G-II:  TA ended ~May 27, 2025  -> Jan 1 2026 = 219 days
# G-III: TA ended ~Mar 24, 2025  -> Jan 1 2026 = 283 days
_CATALYST_START_AGE: dict[str, int] = {
    "gadepan_1": 247,
    "gadepan_2": 219,
    "gadepan_3": 283,
}

# Per-plant DCS measurement noise on shared environmental signals
_DCS_MEASUREMENT_NOISE: dict[str, float] = {
    "ambient_temp":      0.3,
    "ambient_humidity":  1.0,
    "ng_calorific_value": 8.0,
}

_MISSING_RATE: float = 0.002  # 0.2 % sensor dropout rate


# =====================================================================
# Generator Class
# =====================================================================

class SyntheticDataGenerator:
    """Generates realistic synthetic DCS data for the CFCL SEC pipeline.

    Parameters
    ----------
    config_dir : Path, optional
        Override the config directory path.
    seed : int
        Random seed for reproducibility (default 42).
    """

    def __init__(
        self,
        config_dir: Optional[Path] = None,
        seed: int = 42,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.plant_cfg = load_config("plant", config_dir=config_dir)
        self.plants: dict = self.plant_cfg["plants"]
        self.pv_specs: dict = self.plant_cfg["process_variables"]
        self.data_gen: dict = self.plant_cfg["data_generation"]
        self._log = get_logger(__name__, file_logging=False)

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def generate(self) -> pd.DataFrame:
        """Generate synthetic data for all 3 plants.

        Environmental signals are generated once and shared across plants.
        Turnaround periods are excluded (plant offline for maintenance).

        Returns
        -------
        pd.DataFrame
            Columns match ``RAW_COLUMNS`` from ``src.data.schema``.
        """
        timestamps = self._create_timestamps()
        n = len(timestamps)
        hours = timestamps.hour.values
        months = timestamps.month.values

        # -- Shared environmental signals (same weather / gas for all) --
        shared_ambient_temp = self._generate_ambient_temp(n, hours, months)
        shared_ambient_humidity = self._generate_ambient_humidity(
            n, hours, months
        )
        ng_spec = self.pv_specs["ng_calorific_value"]
        ng_typ = ng_spec.get("typical_range", [8600, 9000])
        shared_ng_cv = np.clip(
            self._ar1_series(
                n,
                mean=(ng_typ[0] + ng_typ[1]) / 2,
                std=(ng_typ[1] - ng_typ[0]) / 3,
            ),
            float(ng_spec["safety_min"]),
            float(ng_spec["safety_max"]),
        )

        shared_env = {
            "ambient_temp": shared_ambient_temp,
            "ambient_humidity": shared_ambient_humidity,
            "ng_calorific_value": shared_ng_cv,
        }

        self._log.info(
            "Shared env: temp %.1f+/-%.1f, hum %.1f+/-%.1f, NG %.0f+/-%.0f",
            shared_ambient_temp.mean(), shared_ambient_temp.std(),
            shared_ambient_humidity.mean(), shared_ambient_humidity.std(),
            shared_ng_cv.mean(), shared_ng_cv.std(),
        )

        # -- Per-plant data generation --------------------------------
        frames: list[pd.DataFrame] = []

        for plant_id, pcfg in self.plants.items():
            self._log.info("Generating data for %s ...", pcfg["name"])
            df = self._generate_plant(
                plant_id, pcfg, timestamps, shared_env
            )
            frames.append(df)
            self._log.info(
                "  %s: %d records, SEC %.3f +/- %.3f",
                pcfg["name"], len(df),
                df["sec_gcal_per_mt"].mean(),
                df["sec_gcal_per_mt"].std(),
            )

        result = (
            pd.concat(frames, ignore_index=True)
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        self._log.info(
            "Total: %d records across %d plants", len(result), len(frames)
        )
        return result

    def save(
        self,
        df: pd.DataFrame,
        filename: str = "plant_data_raw.csv",
    ) -> Path:
        """Save generated data to ``data/raw/``."""
        output_dir = get_project_root() / "data" / "raw"
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / filename
        df.to_csv(filepath, index=False)
        self._log.info("Saved %d records to %s", len(df), filepath)
        return filepath

    # -----------------------------------------------------------------
    # Per-Plant Generation
    # -----------------------------------------------------------------

    def _generate_plant(
        self,
        plant_id: str,
        pcfg: dict,
        timestamps: pd.DatetimeIndex,
        shared_env: dict[str, np.ndarray],
    ) -> pd.DataFrame:
        """Generate full dataset for a single plant."""
        n = len(timestamps)
        hours = timestamps.hour.values
        months = timestamps.month.values
        weekdays = timestamps.weekday.values

        # 1. Determine turnaround window
        ta_start, ta_end = self._get_turnaround_window(plant_id, pcfg)

        # 2. Process variables (shared env + per-plant controllables)
        variables = self._generate_variables(
            plant_id, pcfg, timestamps, shared_env
        )

        # 3. Catalyst age (with turnaround reset)
        variables["catalyst_age_days"] = self._calculate_catalyst_age(
            plant_id, timestamps, ta_end
        )

        # 4. Calculate SEC
        sec = self._calculate_sec(variables, pcfg, hours, months, weekdays)

        # 5. Inject anomalies
        n_anomalies = self.data_gen.get("anomaly_count", 3)
        sec, _ = self._inject_anomalies(sec, timestamps, n_anomalies, plant_id)

        # 6. Assign shifts
        shifts = self._assign_shifts(hours)

        # 7. Assemble DataFrame
        data: dict[str, object] = {
            "timestamp": timestamps,
            "plant_id": plant_id,
        }
        for var_name in self.pv_specs:
            data[var_name] = variables[var_name]
        data["shift"] = shifts
        data["sec_gcal_per_mt"] = np.round(sec, 4)

        df = pd.DataFrame(data)

        # 8. Remove turnaround period (plant offline)
        if ta_start is not None:
            before = len(df)
            mask = ~((df["timestamp"] >= ta_start) & (df["timestamp"] < ta_end))
            df = df[mask].reset_index(drop=True)
            removed = before - len(df)
            self._log.info(
                "  Turnaround: %s to %s (%d hours removed, plant offline)",
                ta_start.strftime("%b %d"),
                ta_end.strftime("%b %d"),
                removed,
            )

        # 9. Inject sparse missing values
        df = self._inject_missing_values(df)

        return df

    # -----------------------------------------------------------------
    # Turnaround Window
    # -----------------------------------------------------------------

    def _get_turnaround_window(
        self, plant_id: str, pcfg: dict
    ) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """Compute turnaround start/end dates from plant config.

        Returns (ta_start, ta_end) or (None, None) if no TA in range.
        """
        ta = pcfg.get("turnaround", {})
        months = ta.get("typical_months", [])
        duration = ta.get("duration_days", 25)
        if not months:
            return None, None

        start_year = pd.Timestamp(self.data_gen["start_date"]).year
        data_start = pd.Timestamp(self.data_gen["start_date"])
        data_end = data_start + pd.DateOffset(
            months=self.data_gen["duration_months"]
        )

        # TA starts on the 5th of the first typical month
        ta_start = pd.Timestamp(year=start_year, month=months[0], day=5)
        ta_end = ta_start + pd.Timedelta(days=duration)

        # Only include if TA falls within data window
        if ta_start >= data_start and ta_start < data_end:
            return ta_start, ta_end
        return None, None

    # -----------------------------------------------------------------
    # Catalyst Age (with Turnaround Reset)
    # -----------------------------------------------------------------

    def _calculate_catalyst_age(
        self,
        plant_id: str,
        timestamps: pd.DatetimeIndex,
        ta_end: Optional[pd.Timestamp],
    ) -> np.ndarray:
        """Calculate catalyst age with reset after turnaround.

        Before turnaround: age = start_age + days since Jan 1.
        After turnaround:  age = days since turnaround completion.
        """
        start_age = _CATALYST_START_AGE.get(plant_id, 200)
        data_start = timestamps[0]
        day_offsets = (timestamps - data_start).days.values

        if ta_end is not None:
            ta_end_day = (ta_end - data_start).days
            pre_ta = timestamps < ta_end
            # Before TA: continuing from last year's age
            # After TA: fresh catalyst, counting from 0
            catalyst_age = np.where(
                pre_ta,
                start_age + day_offsets,
                day_offsets - ta_end_day,
            )
        else:
            catalyst_age = start_age + day_offsets

        return catalyst_age.astype(int)

    # -----------------------------------------------------------------
    # Timestamp Generation
    # -----------------------------------------------------------------

    def _create_timestamps(self) -> pd.DatetimeIndex:
        """Create hourly timestamp range from data_generation config."""
        start = pd.Timestamp(self.data_gen["start_date"])
        end = start + pd.DateOffset(months=self.data_gen["duration_months"])
        return pd.date_range(
            start=start, end=end,
            freq=self.data_gen["frequency"],
            inclusive="left",
        )

    # -----------------------------------------------------------------
    # Process Variable Generation
    # -----------------------------------------------------------------

    def _generate_variables(
        self,
        plant_id: str,
        pcfg: dict,
        timestamps: pd.DatetimeIndex,
        shared_env: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Generate all process variables with realistic temporal patterns."""
        n = len(timestamps)
        control = _CONTROL_TIGHTNESS.get(plant_id, 1.0)
        variables: dict[str, np.ndarray] = {}

        for var_name, spec in self.pv_specs.items():
            if var_name in variables or var_name == "catalyst_age_days":
                continue

            smin = float(spec["safety_min"])
            smax = float(spec["safety_max"])

            # -- Shared environmental variables --
            if var_name in shared_env:
                base_signal = shared_env[var_name].copy()
                noise_std = _DCS_MEASUREMENT_NOISE.get(var_name, 0.0)
                if noise_std > 0:
                    base_signal += self.rng.normal(0, noise_std, n)
                variables[var_name] = np.clip(base_signal, smin, smax)

            # -- CW inlet temp (correlated with shared ambient) --
            elif var_name == "cw_inlet_temp":
                if "ambient_temp" not in variables:
                    ambient = shared_env["ambient_temp"]
                else:
                    ambient = variables["ambient_temp"]
                cw_base = 0.40 * ambient + 16.0
                cw_noise = self.rng.normal(0, 1.0 * control, n)
                variables[var_name] = np.clip(cw_base + cw_noise, smin, smax)

            # -- CO2 slip (plant-specific technology) --
            elif var_name == "co2_slip_ppm":
                slip_range = pcfg.get("co2_slip", {}).get(
                    "range", [smin, smax]
                )
                slip_mean = (slip_range[0] + slip_range[1]) / 2
                slip_std = (slip_range[1] - slip_range[0]) / 5 * control
                variables[var_name] = np.clip(
                    self._ar1_series(n, slip_mean, slip_std), smin, smax
                )

            # -- All other variables (AR(1) around optimal) --
            else:
                if "optimal_range" in spec:
                    opt = spec["optimal_range"]
                    mean = (opt[0] + opt[1]) / 2
                    std = (opt[1] - opt[0]) / 3 * control
                elif "typical_range" in spec:
                    typ = spec["typical_range"]
                    mean = (typ[0] + typ[1]) / 2
                    std = (typ[1] - typ[0]) / 3 * control
                else:
                    mean = (smin + smax) / 2
                    std = (smax - smin) / 6 * control

                variables[var_name] = np.clip(
                    self._ar1_series(n, mean, std), smin, smax
                )

        return variables

    # -----------------------------------------------------------------
    # Ambient Variable Generators (Patterns 1 & 2)
    # -----------------------------------------------------------------

    def _generate_ambient_temp(
        self, n: int, hours: np.ndarray, months: np.ndarray
    ) -> np.ndarray:
        """Ambient temperature with diurnal + seasonal patterns."""
        temp_means = np.array([
            _KOTA_CLIMATE.get(m, _KOTA_CLIMATE[1])["temp_mean"]
            for m in months
        ])
        temp_amps = np.array([
            _KOTA_CLIMATE.get(m, _KOTA_CLIMATE[1])["temp_amp"]
            for m in months
        ])
        diurnal = temp_amps * np.sin(2 * np.pi * (hours - 9) / 24)
        noise = self.rng.normal(0, 1.8, n)
        temp = temp_means + diurnal + noise

        for i in range(1, n):
            temp[i] = 0.70 * temp[i] + 0.30 * temp[i - 1]

        return np.clip(temp, 12.0, 48.0)

    def _generate_ambient_humidity(
        self, n: int, hours: np.ndarray, months: np.ndarray
    ) -> np.ndarray:
        """Ambient humidity with seasonal pattern."""
        hum_means = np.array([
            _KOTA_CLIMATE.get(m, _KOTA_CLIMATE[1])["hum_mean"]
            for m in months
        ])
        hum_amps = np.array([
            _KOTA_CLIMATE.get(m, _KOTA_CLIMATE[1])["hum_amp"]
            for m in months
        ])
        diurnal = -hum_amps * np.sin(2 * np.pi * (hours - 9) / 24)
        noise = self.rng.normal(0, 4.0, n)
        hum = hum_means + diurnal + noise

        for i in range(1, n):
            hum[i] = 0.70 * hum[i] + 0.30 * hum[i - 1]

        return np.clip(hum, 10.0, 100.0)

    # -----------------------------------------------------------------
    # AR(1) Time-Series Helper
    # -----------------------------------------------------------------

    def _ar1_series(
        self, n: int, mean: float, std: float, alpha: float = 0.97,
    ) -> np.ndarray:
        """Generate an AR(1) autocorrelated time series."""
        innovation_std = std * np.sqrt(1.0 - alpha ** 2)
        x = np.empty(n)
        x[0] = mean + self.rng.normal(0, std)
        for i in range(1, n):
            x[i] = mean + alpha * (x[i - 1] - mean) + self.rng.normal(
                0, innovation_std
            )
        return x

    # -----------------------------------------------------------------
    # SEC Calculation (Patterns 1-5, 7)
    # -----------------------------------------------------------------

    def _calculate_sec(
        self,
        variables: dict[str, np.ndarray],
        pcfg: dict,
        hours: np.ndarray,
        months: np.ndarray,
        weekdays: np.ndarray,
    ) -> np.ndarray:
        """Calculate SEC from process variables and temporal effects."""
        n = len(hours)
        benchmark = float(pcfg["sec_range"]["benchmark"])
        sec = np.full(n, benchmark)

        # -- Linear contributions --
        for var_name, coeff in _SEC_LINEAR.items():
            spec = self.pv_specs[var_name]
            smin = float(spec["safety_min"])
            smax = float(spec["safety_max"])
            var_range = smax - smin

            if var_name == "co2_slip_ppm":
                slip_range = pcfg.get("co2_slip", {}).get(
                    "range", [smin, smax]
                )
                centre = (slip_range[0] + slip_range[1]) / 2
            elif "optimal_range" in spec:
                centre = sum(spec["optimal_range"]) / 2
            elif "typical_range" in spec:
                centre = sum(spec["typical_range"]) / 2
            else:
                centre = (smin + smax) / 2

            norm_dev = (variables[var_name] - centre) / var_range
            sec += coeff * norm_dev

        # -- Nonlinear: steam-carbon x reformer temp --
        sc_range = (self.pv_specs["steam_carbon_ratio"]["safety_max"]
                    - self.pv_specs["steam_carbon_ratio"]["safety_min"])
        rt_range = (self.pv_specs["reformer_outlet_temp"]["safety_max"]
                    - self.pv_specs["reformer_outlet_temp"]["safety_min"])
        sc_opt = sum(self.pv_specs["steam_carbon_ratio"]["optimal_range"]) / 2
        rt_opt = sum(self.pv_specs["reformer_outlet_temp"]["optimal_range"]) / 2
        sc_norm = (variables["steam_carbon_ratio"] - sc_opt) / sc_range
        rt_norm = (variables["reformer_outlet_temp"] - rt_opt) / rt_range
        sec += 0.04 * sc_norm * rt_norm

        # -- Nonlinear: ambient temp x CW inlet --
        at_range = (self.pv_specs["ambient_temp"]["safety_max"]
                    - self.pv_specs["ambient_temp"]["safety_min"])
        cw_range = (self.pv_specs["cw_inlet_temp"]["safety_max"]
                    - self.pv_specs["cw_inlet_temp"]["safety_min"])
        at_norm = (variables["ambient_temp"] - 28.0) / at_range
        cw_norm = (variables["cw_inlet_temp"] - 30.0) / cw_range
        sec += 0.04 * at_norm * cw_norm

        # -- Humidity threshold penalty --
        humidity = variables["ambient_humidity"]
        sec += 0.001 * np.maximum(0.0, humidity - 55.0) ** 1.3

        # -- Catalyst aging: exponential decay --
        age = np.maximum(variables["catalyst_age_days"].astype(float), 0)
        sec += 0.08 * (1.0 - np.exp(-age / 300.0))

        # -- Night shift drift --
        is_night = ((hours >= 22) | (hours < 6)).astype(float)
        sec += _NIGHT_SHIFT_PENALTY * is_night

        # -- Noise (tighter on weekends) --
        is_weekend = ((weekdays == 5) | (weekdays == 6)).astype(float)
        noise_scale = _NOISE_STD * (
            1.0 - (1.0 - _WEEKEND_NOISE_FACTOR) * is_weekend
        )
        sec += self.rng.normal(0, 1, n) * noise_scale

        # -- Clip --
        sec_min = float(pcfg["sec_range"]["min"]) - 0.05
        sec_max = float(pcfg["sec_range"]["max"]) + 0.20
        return np.clip(sec, sec_min, sec_max)

    # -----------------------------------------------------------------
    # Anomaly Injection (Pattern 6)
    # -----------------------------------------------------------------

    def _inject_anomalies(
        self, sec: np.ndarray, timestamps: pd.DatetimeIndex,
        n_anomalies: int, plant_id: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Inject compressor trip anomalies (2-4h SEC spikes)."""
        n = len(sec)
        anomaly_mask = np.zeros(n, dtype=bool)
        if n_anomalies <= 0:
            return sec, anomaly_mask

        margin = 72
        usable = n - 2 * margin
        segment = usable // n_anomalies

        for i in range(n_anomalies):
            seg_start = margin + i * segment
            seg_end = margin + (i + 1) * segment
            start_idx = int(
                self.rng.integers(seg_start, min(seg_end, n - margin))
            )
            duration = int(self.rng.integers(2, 5))
            end_idx = min(start_idx + duration, n)
            peak = self.rng.uniform(0.30, 0.60)

            for j in range(start_idx, end_idx):
                decay = 1.0 - 0.6 * (j - start_idx) / duration
                sec[j] += peak * decay
                anomaly_mask[j] = True

            self._log.info(
                "  Anomaly @ %s (%dh, +%.2f Gcal/MT) -- %s",
                timestamps[start_idx].strftime("%Y-%m-%d %H:%M"),
                duration, peak, plant_id,
            )

        return sec, anomaly_mask

    # -----------------------------------------------------------------
    # Missing Values & Shift Assignment
    # -----------------------------------------------------------------

    def _inject_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject sparse NaN values (~0.2%) for DCS sensor dropouts."""
        numeric_cols = [
            c for c in df.columns
            if c not in ("timestamp", "plant_id", "shift")
        ]
        mask = self.rng.random((len(df), len(numeric_cols))) < _MISSING_RATE
        for idx, col in enumerate(numeric_cols):
            df.loc[mask[:, idx], col] = np.nan
        n_missing = int(mask.sum())
        if n_missing > 0:
            self._log.info(
                "  Injected %d missing values (%.2f %%)",
                n_missing, 100 * n_missing / mask.size,
            )
        return df

    @staticmethod
    def _assign_shifts(hours: np.ndarray) -> np.ndarray:
        """Map hour-of-day to operator shift labels."""
        return np.where(
            (hours >= 6) & (hours < 14), "morning",
            np.where(
                (hours >= 14) & (hours < 22), "afternoon",
                "night",
            ),
        )


# =====================================================================
# CLI Entry Point
# =====================================================================

def main() -> None:
    """Generate synthetic DCS data and save to data/raw/."""
    generator = SyntheticDataGenerator()
    df = generator.generate()
    filepath = generator.save(df)

    print(f"\n{'=' * 60}")
    print(f"Generated {len(df):,} records for {df['plant_id'].nunique()} plants")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Saved to:   {filepath}\n")

    print("Per-plant summary:")
    for pid in sorted(df["plant_id"].unique()):
        p = df[df["plant_id"] == pid]
        cat = p["catalyst_age_days"].dropna()
        print(f"  {pid}: {len(p):,} records, "
              f"SEC {p['sec_gcal_per_mt'].mean():.3f} +/- {p['sec_gcal_per_mt'].std():.3f}, "
              f"catalyst {int(cat.min())}-{int(cat.max())} days")


if __name__ == "__main__":
    main()
