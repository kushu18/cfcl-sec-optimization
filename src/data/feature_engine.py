"""Feature engineering module for the CFCL SEC Optimization pipeline.

Transforms preprocessed DCS data into a rich feature matrix for ML models.
Engineered features capture domain-specific process interactions, temporal
dynamics, catalyst degradation curves, operator-controllable deviations
from optimal setpoints, and cross-plant fleet-level comparisons.

Feature groups (total ~160 features):
    - Rolling statistics (3h/6h/12h/24h windows)  — 72
    - Exponential weighted moving averages (EWMA)  — 27
    - Process interaction terms                    — 5
    - Deviation from optimal setpoints             — 6
    - SEC lag / diff features                      — 6
    - Rate-of-change (first derivative) features   — 9
    - Catalyst degradation features                — 7
    - Efficiency ratio cross-metrics               — 3
    - Cross-plant relative features                — 4
    - Cumulative deviation tracking                — 3
    - Operating regime classification              — 5
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.data.schema import (
    CONTROLLABLE_FEATURES,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from src.utils.config_loader import get_project_root, load_config
from src.utils.logger import get_logger


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ROLLING_WINDOWS: list[int] = [3, 6, 12, 24]  # hours
_EWMA_SPANS: list[int] = [6, 12, 24]  # hours (exponential decay half-lives)

_INTERACTION_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "steam_reformer_interaction",
        "op": "multiply",
        "cols": ("steam_carbon_ratio", "reformer_outlet_temp"),
        "desc": "Steam injection vs. reformer exit temperature coupling",
    },
    {
        "name": "ambient_cw_interaction",
        "op": "multiply",
        "cols": ("ambient_temp", "cw_inlet_temp"),
        "desc": "Ambient heat load on cooling water performance",
    },
    {
        "name": "pressure_temp_ratio",
        "op": "divide",
        "cols": ("loop_pressure", "reformer_outlet_temp"),
        "desc": "Synthesis pressure per unit reformer temperature",
    },
    {
        "name": "humidity_cw_interaction",
        "op": "multiply",
        "cols": ("ambient_humidity", "cw_inlet_temp"),
        "desc": "Humidity impact on cooling tower evaporative efficiency",
    },
    {
        "name": "steam_pressure_interaction",
        "op": "multiply",
        "cols": ("steam_carbon_ratio", "loop_pressure"),
        "desc": "Steam injection vs. synthesis loop pressure coupling",
    },
]

_LAG_HOURS: list[int] = [1, 3, 6, 12, 24]

_CATALYST_AGE_BINS: list[int] = [0, 30, 90, 180, 365, 2500]
_CATALYST_AGE_LABELS: list[str] = ["fresh", "young", "middle", "aged", "old"]

# Features for which rate-of-change (first derivative) is meaningful
_RATE_OF_CHANGE_FEATURES: list[str] = FEATURE_COLUMNS  # all 9 raw features

# Operating regime classification thresholds
_STEADY_STATE_SEC_STD_THRESHOLD = 0.02  # 6h rolling std < 0.02 = steady
_TRANSIENT_SEC_STD_THRESHOLD = 0.05     # > 0.05 = transient / upset


class FeatureEngine:
    """Domain-aware feature engineering for SEC prediction models.

    Generates ~160 features from 9 raw process variables, including rolling
    statistics, EWMA, process interactions, optimal-setpoint deviations,
    SEC autoregressive lags, rate-of-change derivatives, catalyst
    degradation transforms, efficiency ratios, cross-plant relative
    metrics, cumulative deviation tracking, and operating regime flags.

    Parameters
    ----------
    config_name : str
        Config key for :func:`load_config` (default ``"plant"``).
    logger : logging.Logger, optional
        Custom logger; a module-level default is created if ``None``.
    """

    def __init__(
        self,
        config_name: str = "plant",
        logger: Optional[Any] = None,
    ) -> None:
        self.logger = logger or get_logger("feature_engine")
        self.plant_cfg = load_config(config_name)
        self._feature_names: list[str] = []
        self._metadata: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 1. Rolling statistics
    # ------------------------------------------------------------------

    def compute_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling mean and std for each numeric feature per plant.

        Windows: 3h, 6h, 12h, 24h.  ``min_periods=1`` avoids NaN at
        the start of each plant's timeseries.

        Parameters
        ----------
        df : pd.DataFrame
            Must be sorted by ``(plant_id, timestamp)``.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with ``9 x 4 x 2 = 72`` new rolling columns.
        """
        self.logger.info(
            "Computing rolling statistics (windows: %s) for %d features",
            _ROLLING_WINDOWS, len(FEATURE_COLUMNS),
        )

        rolling_cols: list[str] = []

        for plant_id, grp in df.groupby("plant_id", sort=False):
            grp = grp.sort_values("timestamp")
            idx = grp.index

            for feat in FEATURE_COLUMNS:
                series = grp[feat]
                for window in _ROLLING_WINDOWS:
                    mean_col = f"{feat}_rolling_mean_{window}h"
                    std_col = f"{feat}_rolling_std_{window}h"

                    rolling = series.rolling(window=window, min_periods=1)
                    df.loc[idx, mean_col] = rolling.mean().values
                    df.loc[idx, std_col] = rolling.std(ddof=0).values

                    if mean_col not in rolling_cols:
                        rolling_cols.extend([mean_col, std_col])

        self._metadata["rolling_features"] = rolling_cols
        self.logger.info("  Added %d rolling features", len(rolling_cols))
        return df

    # ------------------------------------------------------------------
    # 2. Exponential Weighted Moving Averages (EWMA)
    # ------------------------------------------------------------------

    def compute_ewma_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute EWMA for all process features per plant.

        EWMA places exponentially decaying weight on older observations,
        making it more responsive to recent process changes than simple
        rolling averages.  Critical for capturing gradual catalyst
        degradation and ambient temperature trends.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Input DataFrame with ``9 x 3 = 27`` EWMA columns.
        """
        self.logger.info(
            "Computing EWMA features (spans: %s) for %d features",
            _EWMA_SPANS, len(FEATURE_COLUMNS),
        )
        ewma_cols: list[str] = []

        for plant_id, grp in df.groupby("plant_id", sort=False):
            grp = grp.sort_values("timestamp")
            idx = grp.index

            for feat in FEATURE_COLUMNS:
                series = grp[feat]
                for span in _EWMA_SPANS:
                    col_name = f"{feat}_ewma_{span}h"
                    df.loc[idx, col_name] = (
                        series.ewm(span=span, min_periods=1).mean().values
                    )
                    if col_name not in ewma_cols:
                        ewma_cols.append(col_name)

        self._metadata["ewma_features"] = ewma_cols
        self.logger.info("  Added %d EWMA features", len(ewma_cols))
        return df

    # ------------------------------------------------------------------
    # 3. Interaction terms
    # ------------------------------------------------------------------

    def compute_interaction_terms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create domain-knowledge interaction features.

        Captures known physical couplings between process variables
        (e.g., steam-reformer temperature interaction drives conversion
        efficiency, ambient-CW coupling determines condenser duty).

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Input DataFrame with 5 interaction columns added.
        """
        self.logger.info("Computing %d interaction terms", len(_INTERACTION_DEFINITIONS))
        interaction_cols: list[str] = []

        for defn in _INTERACTION_DEFINITIONS:
            col_a, col_b = defn["cols"]
            name = defn["name"]

            if defn["op"] == "multiply":
                df[name] = df[col_a] * df[col_b]
            elif defn["op"] == "divide":
                df[name] = df[col_a] / df[col_b].replace(0, np.nan)
            else:
                raise ValueError(f"Unknown interaction op: {defn['op']}")

            interaction_cols.append(name)
            self.logger.info("  %s = %s %s %s", name, col_a, defn["op"], col_b)

        self._metadata["interaction_features"] = interaction_cols
        return df

    # ------------------------------------------------------------------
    # 4. Deviation from optimal setpoints
    # ------------------------------------------------------------------

    def compute_deviation_from_optimal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute deviations from operator-optimal setpoint ranges.

        For each controllable feature (steam-to-carbon ratio, loop
        pressure, reformer outlet temperature), calculates:

        - Numeric deviation from the midpoint of the optimal range
        - Binary flag indicating the current value lies outside the
          recommended operating envelope

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Input DataFrame with 6 deviation columns added.
        """
        self.logger.info("Computing deviations from optimal setpoints")
        proc_vars = self.plant_cfg.get("process_variables", {})
        deviation_cols: list[str] = []

        for feat in CONTROLLABLE_FEATURES:
            var_cfg = proc_vars.get(feat, {})
            optimal = var_cfg.get("optimal_range")

            if optimal is None or len(optimal) != 2:
                self.logger.warning(
                    "  %s: no optimal_range in config, skipping", feat
                )
                continue

            lo, hi = optimal
            midpoint = (lo + hi) / 2.0

            dev_col = f"{feat}_deviation"
            flag_col = f"{feat}_outside_optimal"

            df[dev_col] = df[feat] - midpoint
            df[flag_col] = ((df[feat] < lo) | (df[feat] > hi)).astype(int)

            deviation_cols.extend([dev_col, flag_col])
            self.logger.info(
                "  %s: optimal [%.2f, %.2f], midpoint %.2f", feat, lo, hi, midpoint
            )

        self._metadata["deviation_features"] = deviation_cols
        self.logger.info("  Added %d deviation features", len(deviation_cols))
        return df

    # ------------------------------------------------------------------
    # 5. SEC lag features
    # ------------------------------------------------------------------

    def compute_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create autoregressive lag features for the SEC target.

        SEC exhibits strong temporal autocorrelation (AR(1) alpha ~ 0.97).
        Lag features give tree models and linear models direct access
        to this sequential dependency.

        Parameters
        ----------
        df : pd.DataFrame
            Must be sorted by ``(plant_id, timestamp)``.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with 6 lag/diff columns added.
        """
        self.logger.info("Computing SEC lag features (lags: %s)", _LAG_HOURS)
        lag_cols: list[str] = []

        for plant_id, grp in df.groupby("plant_id", sort=False):
            grp = grp.sort_values("timestamp")
            idx = grp.index
            sec = grp[TARGET_COLUMN]

            for lag in _LAG_HOURS:
                lag_col = f"sec_lag_{lag}h"
                df.loc[idx, lag_col] = sec.shift(lag).values
                if lag_col not in lag_cols:
                    lag_cols.append(lag_col)

        # SEC first difference between previous two hours (prevents target leakage)
        diff_col = "sec_diff_1h"
        for plant_id, grp in df.groupby("plant_id", sort=False):
            grp = grp.sort_values("timestamp")
            idx = grp.index
            df.loc[idx, diff_col] = grp[TARGET_COLUMN].shift(1).diff().values

        lag_cols.append(diff_col)

        self._metadata["lag_features"] = lag_cols
        self.logger.info("  Added %d lag/diff features", len(lag_cols))
        return df

    # ------------------------------------------------------------------
    # 6. Rate-of-change (first derivative) features
    # ------------------------------------------------------------------

    def compute_rate_of_change(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute hourly rate-of-change for all process variables.

        In a large-scale urea plant, sudden changes in process variables
        (e.g., rapid reformer temperature swing) are more informative for
        SEC deviation than absolute values alone.  The first difference
        captures this process dynamics signal.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Input DataFrame with 9 rate-of-change columns added.
        """
        self.logger.info("Computing rate-of-change features for %d variables",
                         len(_RATE_OF_CHANGE_FEATURES))
        roc_cols: list[str] = []

        for plant_id, grp in df.groupby("plant_id", sort=False):
            grp = grp.sort_values("timestamp")
            idx = grp.index

            for feat in _RATE_OF_CHANGE_FEATURES:
                roc_col = f"{feat}_roc"
                df.loc[idx, roc_col] = grp[feat].diff().values
                if roc_col not in roc_cols:
                    roc_cols.append(roc_col)

        self._metadata["rate_of_change_features"] = roc_cols
        self.logger.info("  Added %d rate-of-change features", len(roc_cols))
        return df

    # ------------------------------------------------------------------
    # 7. Catalyst degradation features
    # ------------------------------------------------------------------

    def compute_catalyst_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer catalyst-age-derived features.

        Catalyst deactivation follows an exponential decay curve.
        Polynomial and log transforms help models capture the
        non-linear SEC increase as the catalyst ages.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Input DataFrame with ~7 catalyst feature columns added.
        """
        self.logger.info("Computing catalyst degradation features")
        cat_cols: list[str] = []

        # Polynomial transforms
        df["catalyst_age_squared"] = df["catalyst_age_days"] ** 2
        cat_cols.append("catalyst_age_squared")

        # Log transform (+1 to avoid log(0) for fresh catalyst)
        df["catalyst_age_log"] = np.log1p(df["catalyst_age_days"])
        cat_cols.append("catalyst_age_log")

        # Age buckets for categorical segmentation
        df["catalyst_age_bucket"] = pd.cut(
            df["catalyst_age_days"],
            bins=_CATALYST_AGE_BINS,
            labels=_CATALYST_AGE_LABELS,
            include_lowest=True,
            right=True,
        )

        # One-hot encode the buckets
        bucket_dummies = pd.get_dummies(
            df["catalyst_age_bucket"], prefix="catalyst", dtype=int
        )
        for label in _CATALYST_AGE_LABELS:
            col = f"catalyst_{label}"
            if col not in bucket_dummies.columns:
                bucket_dummies[col] = 0

        df = pd.concat([df, bucket_dummies], axis=1)
        cat_cols.extend([f"catalyst_{label}" for label in _CATALYST_AGE_LABELS])

        # Drop the intermediate string bucket column (not numeric)
        df.drop(columns=["catalyst_age_bucket"], inplace=True)

        self._metadata["catalyst_features"] = cat_cols
        self.logger.info("  Added %d catalyst features", len(cat_cols))
        return df

    # ------------------------------------------------------------------
    # 8. Efficiency ratios
    # ------------------------------------------------------------------

    def compute_efficiency_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute cross-variable efficiency ratio features.

        These domain-specific ratios capture thermodynamic relationships
        that single features cannot express.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Input DataFrame with 3 efficiency ratio columns added.
        """
        self.logger.info("Computing efficiency ratio features")
        ratio_cols: list[str] = []

        # Energy intensity: total energy input proxy (based on previous hour)
        df["energy_intensity"] = df["sec_lag_1h"] * df["ng_calorific_value"]
        ratio_cols.append("energy_intensity")

        # CO2 capture efficiency relative to gas quality
        df["co2_efficiency"] = df["co2_slip_ppm"] / df["ng_calorific_value"]
        ratio_cols.append("co2_efficiency")

        # Thermal load index: temperature differential normalised by reformer
        df["thermal_load_index"] = (
            (df["ambient_temp"] - df["cw_inlet_temp"]) / df["reformer_outlet_temp"]
        )
        ratio_cols.append("thermal_load_index")

        self._metadata["efficiency_features"] = ratio_cols
        self.logger.info("  Added %d efficiency ratio features", len(ratio_cols))
        return df

    # ------------------------------------------------------------------
    # 9. Cross-plant relative features
    # ------------------------------------------------------------------

    def compute_cross_plant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute fleet-relative performance features.

        Compares each plant's SEC performance against the fleet average
        using the previous hour's observations (t-1) to eliminate contemporaneous
        target leakage while preserving fleet-wide operational dynamics.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Input DataFrame with 4 cross-plant relative columns.
        """
        self.logger.info("Computing cross-plant relative features (zero target leakage)")
        cross_cols: list[str] = []

        # Fleet-level SEC at timestamp t-1 (via sec_lag_1h)
        fleet_sec_mean = df.groupby("timestamp")["sec_lag_1h"].transform("mean")
        fleet_sec_std = df.groupby("timestamp")["sec_lag_1h"].transform("std").fillna(0)

        # SEC deviation from fleet mean (positive = worse than fleet)
        df["sec_vs_fleet_mean"] = df["sec_lag_1h"] - fleet_sec_mean
        cross_cols.append("sec_vs_fleet_mean")

        # SEC Z-score vs fleet (how many fleet-stds away)
        df["sec_fleet_zscore"] = np.where(
            fleet_sec_std > 0,
            (df["sec_lag_1h"] - fleet_sec_mean) / fleet_sec_std,
            0.0,
        )
        cross_cols.append("sec_fleet_zscore")

        # Relative SEC rank within fleet (0=best, 2=worst at each timestamp)
        df["sec_fleet_rank"] = df.groupby("timestamp")["sec_lag_1h"].rank(
            method="min", ascending=True
        ) - 1
        cross_cols.append("sec_fleet_rank")

        # Fleet range (max - min SEC across all plants at this timestamp)
        fleet_range = (
            df.groupby("timestamp")["sec_lag_1h"].transform("max")
            - df.groupby("timestamp")["sec_lag_1h"].transform("min")
        )
        df["sec_fleet_spread"] = fleet_range
        cross_cols.append("sec_fleet_spread")

        self._metadata["cross_plant_features"] = cross_cols
        self.logger.info("  Added %d cross-plant features", len(cross_cols))
        return df

    # ------------------------------------------------------------------
    # 10. Cumulative deviation tracking
    # ------------------------------------------------------------------

    def compute_cumulative_deviations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Track how long controllable variables have been outside optimal.

        Persistent off-setpoint operation has a compounding effect on SEC
        that instantaneous deviations don't capture.  These features
        count consecutive hours each variable has been outside its
        optimal band.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Input DataFrame with 3 cumulative deviation columns.
        """
        self.logger.info("Computing cumulative deviation tracking")
        proc_vars = self.plant_cfg.get("process_variables", {})
        cum_cols: list[str] = []

        for feat in CONTROLLABLE_FEATURES:
            var_cfg = proc_vars.get(feat, {})
            optimal = var_cfg.get("optimal_range")
            if optimal is None or len(optimal) != 2:
                continue

            lo, hi = optimal
            col_name = f"{feat}_cum_hours_outside"
            cum_cols.append(col_name)

            for plant_id, grp in df.groupby("plant_id", sort=False):
                grp = grp.sort_values("timestamp")
                idx = grp.index
                outside = ((grp[feat] < lo) | (grp[feat] > hi)).astype(int)

                # Cumulative sum with reset on return to optimal
                cum_hours = []
                count = 0
                for val in outside.values:
                    if val == 1:
                        count += 1
                    else:
                        count = 0
                    cum_hours.append(count)

                df.loc[idx, col_name] = cum_hours

        self._metadata["cumulative_deviation_features"] = cum_cols
        self.logger.info("  Added %d cumulative deviation features", len(cum_cols))
        return df

    # ------------------------------------------------------------------
    # 11. Operating regime classification
    # ------------------------------------------------------------------

    def compute_operating_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify each hour into an operating regime.

        SEC prediction accuracy depends heavily on the plant's operating
        state.  This feature group captures whether the plant is in
        steady-state production, transient upset, post-turnaround
        ramp-up, peak ambient heat stress, or night shift.

        Regime features:

        - ``regime_steady_state`` (bool): 6h SEC rolling std < 0.02
        - ``regime_transient`` (bool): 6h SEC rolling std > 0.05
        - ``regime_post_ta`` (bool): within 72h of turnaround restart
        - ``regime_heat_stress`` (bool): ambient_temp > 40 deg C
        - ``regime_night_ops`` (bool): night shift hours (22:00–06:00)

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Input DataFrame with 5 regime flag columns.
        """
        self.logger.info("Computing operating regime classification")
        regime_cols: list[str] = []

        # Compute 6h rolling SEC std per plant using sec_lag_1h (zero target leakage)
        sec_std_6h_col = "sec_lag_rolling_std_6h"
        for plant_id, grp in df.groupby("plant_id", sort=False):
            grp = grp.sort_values("timestamp")
            idx = grp.index
            df.loc[idx, sec_std_6h_col] = (
                grp["sec_lag_1h"]
                .rolling(window=6, min_periods=1)
                .std(ddof=0)
                .fillna(0)
                .values
            )

        # Steady state: very low SEC variability
        df["regime_steady_state"] = (
            df[sec_std_6h_col] < _STEADY_STATE_SEC_STD_THRESHOLD
        ).astype(int)
        regime_cols.append("regime_steady_state")

        # Transient / upset: high SEC variability
        df["regime_transient"] = (
            df[sec_std_6h_col] > _TRANSIENT_SEC_STD_THRESHOLD
        ).astype(int)
        regime_cols.append("regime_transient")

        # Post-turnaround ramp-up (from preprocessor flag)
        if "is_post_ta_rampup" in df.columns:
            df["regime_post_ta"] = df["is_post_ta_rampup"].astype(int)
        else:
            df["regime_post_ta"] = 0
        regime_cols.append("regime_post_ta")

        # Heat stress: ambient > 40C (Kota summer peak)
        df["regime_heat_stress"] = (df["ambient_temp"] > 40.0).astype(int)
        regime_cols.append("regime_heat_stress")

        # Night operations
        if "is_night_shift" in df.columns:
            df["regime_night_ops"] = df["is_night_shift"]
        elif "hour" in df.columns:
            df["regime_night_ops"] = (
                (df["hour"] >= 22) | (df["hour"] < 6)
            ).astype(int)
        else:
            df["regime_night_ops"] = 0
        regime_cols.append("regime_night_ops")

        self._metadata["regime_features"] = regime_cols
        self.logger.info("  Added %d regime features", len(regime_cols))

        # Log regime distribution
        for col in regime_cols:
            pct = 100.0 * df[col].sum() / len(df)
            self.logger.info("    %s: %.1f%% of records", col, pct)

        return df

    # ------------------------------------------------------------------
    # 12. Feature name registry
    # ------------------------------------------------------------------

    def get_feature_names(self) -> list[str]:
        """Return the ordered list of all feature column names.

        Includes both original numeric features and all engineered
        features.  Excludes the target column, timestamp, and original
        string-valued categorical columns (``plant_id``, ``shift``).

        Returns
        -------
        list[str]

        Raises
        ------
        RuntimeError
            If called before :meth:`run`.
        """
        if not self._feature_names:
            raise RuntimeError(
                "Feature names not available — call run() first."
            )
        return list(self._feature_names)

    # ------------------------------------------------------------------
    # 13. Full pipeline
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute the full feature engineering pipeline.

        Steps:
            1.  Rolling statistics (72 features)
            2.  EWMA features (27 features)
            3.  Interaction terms (5 features)
            4.  Deviation from optimal (6 features)
            5.  SEC lag/diff features (6 features)
            6.  Rate-of-change derivatives (9 features)
            7.  Catalyst degradation features (~7 features)
            8.  Efficiency ratios (3 features)
            9.  Cross-plant relative features (4 features)
            10. Cumulative deviation tracking (3 features)
            11. Operating regime classification (5 features)

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed DataFrame from :class:`DataPreprocessor`.

        Returns
        -------
        pd.DataFrame
            Feature-enriched DataFrame ready for model training.
        """
        self.logger.info("=" * 70)
        self.logger.info("STARTING FEATURE ENGINEERING PIPELINE")
        self.logger.info("=" * 70)

        # Ensure sorted by plant + time for correct rolling/lag computation
        df = df.sort_values(["plant_id", "timestamp"]).reset_index(drop=True)

        # Pre-allocate to avoid fragmentation warnings during column insertion
        df = df.copy()

        df = self.compute_rolling_stats(df)
        df = self.compute_ewma_features(df)
        df = self.compute_interaction_terms(df)
        df = self.compute_deviation_from_optimal(df)
        df = self.compute_lag_features(df)
        df = self.compute_rate_of_change(df)
        df = self.compute_catalyst_features(df)
        df = self.compute_efficiency_ratios(df)
        df = self.compute_cross_plant_features(df)
        df = self.compute_cumulative_deviations(df)
        df = self.compute_operating_regime(df)

        # Build feature name list (everything except metadata columns)
        exclude = {
            "timestamp", "plant_id", "shift", TARGET_COLUMN,
            "catalyst_age_bucket", "outlier_features",
        }
        self._feature_names = [c for c in df.columns if c not in exclude]

        # --- Compile metadata ---
        self._metadata["total_features"] = len(self._feature_names)
        self._metadata["original_features"] = list(FEATURE_COLUMNS)
        self._metadata["feature_groups"] = {
            "original": list(FEATURE_COLUMNS),
            "rolling": self._metadata.get("rolling_features", []),
            "ewma": self._metadata.get("ewma_features", []),
            "interaction": self._metadata.get("interaction_features", []),
            "deviation": self._metadata.get("deviation_features", []),
            "lag": self._metadata.get("lag_features", []),
            "rate_of_change": self._metadata.get("rate_of_change_features", []),
            "catalyst": self._metadata.get("catalyst_features", []),
            "efficiency": self._metadata.get("efficiency_features", []),
            "cross_plant": self._metadata.get("cross_plant_features", []),
            "cumulative_deviation": self._metadata.get("cumulative_deviation_features", []),
            "regime": self._metadata.get("regime_features", []),
        }

        # Count total engineered (non-original) features
        engineered_count = sum(
            len(v) for k, v in self._metadata["feature_groups"].items()
            if k != "original"
        )
        self._metadata["engineered_feature_count"] = engineered_count

        # Save metadata
        output_dir = get_project_root() / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        meta_path = output_dir / "feature_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, indent=2, default=str)
        self.logger.info("Saved feature metadata to %s", meta_path)

        features_csv_path = output_dir / "plant_features.csv"
        df.to_csv(features_csv_path, index=False)
        self.logger.info("Saved feature matrix to %s (%d rows, %d cols)", features_csv_path, len(df), len(df.columns))

        features_parquet_path = output_dir / "plant_features.parquet"
        df.to_parquet(features_parquet_path, index=False, engine="pyarrow", compression="snappy")
        self.logger.info("Saved compressed Parquet telemetry to %s", features_parquet_path)

        nan_count = df[self._feature_names].isna().sum().sum()
        self.logger.info(
            "Feature engineering complete: %d total features (%d engineered), %d NaN values",
            len(self._feature_names), engineered_count, nan_count,
        )
        self.logger.info("=" * 70)
        self.logger.info("FEATURE ENGINEERING COMPLETE")
        self.logger.info("=" * 70)
        return df

    @property
    def metadata(self) -> dict[str, Any]:
        """Return the feature metadata dictionary (populated after :meth:`run`)."""
        return self._metadata


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.data.preprocessor import DataPreprocessor

    # Step 1: Preprocess raw data
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.run()

    # Step 2: Engineer features
    engine = FeatureEngine()
    df_features = engine.run(df_processed)

    print(f"\nFinal DataFrame: {df_features.shape[0]} rows x {df_features.shape[1]} columns")
    print(f"Total features: {engine.metadata.get('total_features', 'N/A')}")
    print(f"Engineered features: {engine.metadata.get('engineered_feature_count', 'N/A')}")
    print(f"\nFeature groups:")
    for group, feats in engine.metadata.get("feature_groups", {}).items():
        print(f"  {group:>25s}: {len(feats):>3} features")
    print(f"\nNaN in features: {df_features[engine.get_feature_names()].isna().sum().sum()}")
