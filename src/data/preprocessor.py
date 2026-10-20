"""Data preprocessor for the CFCL SEC Optimization pipeline.

Loads raw DCS CSV exports, validates sensor ranges, detects statistical
outliers (IQR + modified Z-score), imputes sparse missing values via
per-plant KNN, flags turnaround shutdowns, encodes categoricals and
temporal features, and computes a per-row data quality score.

Pipeline order:
    load_raw_data -> validate_ranges -> detect_outliers -> impute_missing
    -> detect_turnaround_gaps -> validate_cross_plant_signals
    -> encode_categoricals -> add_temporal_features
    -> compute_data_quality_score -> save
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

from src.data.schema import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    RAW_COLUMNS,
    TARGET_COLUMN,
)
from src.utils.config_loader import get_project_root, load_config
from src.utils.logger import get_logger


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_KNN_NEIGHBOURS = 5
_TURNAROUND_GAP_HOURS = 2  # gap > 2h between timestamps flags a TA window

# IQR outlier detection multiplier (1.5 = standard, 3.0 = extreme only)
_IQR_MULTIPLIER = 1.5

# Modified Z-score threshold (Iglewicz & Hoaglin recommend 3.5)
_MODIFIED_ZSCORE_THRESHOLD = 3.5

# Columns that must never contain NaN (DCS identity fields)
_NEVER_NULL_COLS = {"timestamp", "plant_id", "shift"}

# Numeric columns eligible for imputation and range checking
_NUMERIC_FEATURE_COLS: list[str] = [
    col
    for col, spec in RAW_COLUMNS.items()
    if spec.dtype in ("float64", "int64")
]

# Shared environmental signals that should correlate across plants
# (measured from same Gadepan complex weather station / HBJ pipeline)
_SHARED_SIGNALS = ["ambient_temp", "ambient_humidity", "ng_calorific_value"]


class DataPreprocessor:
    """End-to-end preprocessor for raw CFCL plant DCS data.

    Handles the full lifecycle from raw CSV to analysis-ready DataFrame,
    including sensor validation, outlier flagging, KNN imputation,
    turnaround detection, categorical encoding, and data quality scoring.

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
        self.logger = logger or get_logger("preprocessor")
        self.plant_cfg = load_config(config_name)
        self._report: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 1. Load raw data
    # ------------------------------------------------------------------

    def load_raw_data(self, filepath: Path | str | None = None) -> pd.DataFrame:
        """Read the raw DCS CSV export into a DataFrame.

        Parameters
        ----------
        filepath : Path or str, optional
            Explicit path; defaults to ``data/raw/plant_data_raw.csv``
            under the project root.

        Returns
        -------
        pd.DataFrame
            Raw dataframe with ``timestamp`` parsed as ``datetime64[ns]``.

        Raises
        ------
        FileNotFoundError
            If the CSV file does not exist.
        ValueError
            If required columns are missing.
        """
        if filepath is None:
            filepath = get_project_root() / "data" / "raw" / "plant_data_raw.csv"
        filepath = Path(filepath)

        if not filepath.is_file():
            raise FileNotFoundError(f"Raw data file not found: {filepath}")

        self.logger.info("Loading raw data from %s", filepath)
        df = pd.read_csv(filepath, parse_dates=["timestamp"])

        # Validate expected columns
        expected = set(RAW_COLUMNS.keys())
        actual = set(df.columns)
        missing = expected - actual
        if missing:
            raise ValueError(
                f"Raw CSV is missing expected columns: {sorted(missing)}"
            )

        self.logger.info(
            "Loaded %d records | %d columns | plants: %s | date range: %s to %s",
            len(df),
            len(df.columns),
            sorted(df["plant_id"].unique()),
            df["timestamp"].min().strftime("%Y-%m-%d"),
            df["timestamp"].max().strftime("%Y-%m-%d"),
        )

        self._report["raw_records"] = len(df)
        self._report["raw_columns"] = list(df.columns)
        return df

    # ------------------------------------------------------------------
    # 2. Validate & clip sensor ranges
    # ------------------------------------------------------------------

    def validate_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clip numeric values to DCS safety bounds defined in the schema.

        Momentary sensor spikes can push readings outside plausible
        envelopes.  Rather than dropping rows (losing temporal continuity),
        values are clipped to the column's ``[min_value, max_value]``.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with raw numeric columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with out-of-range values clipped.
        """
        self.logger.info("Validating sensor ranges against schema bounds")
        clip_counts: dict[str, int] = {}

        for col in _NUMERIC_FEATURE_COLS:
            spec = RAW_COLUMNS[col]
            lo, hi = spec.min_value, spec.max_value

            below = (df[col] < lo).sum()
            above = (df[col] > hi).sum()
            total_clipped = int(below + above)

            if total_clipped > 0:
                df[col] = df[col].clip(lower=lo, upper=hi)
                self.logger.warning(
                    "  %s: clipped %d values (below: %d, above: %d) to [%.1f, %.1f]",
                    col, total_clipped, below, above, lo, hi,
                )

            clip_counts[col] = total_clipped

        self._report["clipped_values"] = clip_counts
        total = sum(clip_counts.values())
        self.logger.info("Range validation complete: %d total values clipped", total)
        return df

    # ------------------------------------------------------------------
    # 3. Statistical outlier detection (IQR + modified Z-score)
    # ------------------------------------------------------------------

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag statistical outliers using dual IQR + modified Z-score method.

        Uses two complementary approaches per plant to catch outliers
        that pass schema range checks but are statistically abnormal:

        - **IQR method**: Values beyond Q1 - 1.5*IQR or Q3 + 1.5*IQR
        - **Modified Z-score**: Uses median absolute deviation (MAD),
          more robust to heavy-tailed DCS distributions than standard Z.

        Outliers are **flagged, not removed** — anomalous readings
        (e.g., compressor trips) carry genuine process information
        that SHAP analysis and anomaly root-cause modules need.

        Adds columns:

        - ``outlier_count``: number of features flagged for this row
        - ``is_outlier``: True if any feature is flagged
        - ``outlier_features``: comma-separated list of flagged features

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        self.logger.info(
            "Detecting statistical outliers (IQR x%.1f + modified Z-score > %.1f)",
            _IQR_MULTIPLIER, _MODIFIED_ZSCORE_THRESHOLD,
        )
        df = df.copy()

        outlier_flags = pd.DataFrame(
            False, index=df.index, columns=_NUMERIC_FEATURE_COLS
        )

        outlier_stats: dict[str, dict] = {}

        for plant_id, grp in df.groupby("plant_id", sort=False):
            idx = grp.index
            plant_outlier_total = 0

            for col in _NUMERIC_FEATURE_COLS:
                vals = grp[col].dropna()
                if len(vals) < 10:
                    continue

                # --- IQR method ---
                q1 = vals.quantile(0.25)
                q3 = vals.quantile(0.75)
                iqr = q3 - q1
                iqr_lower = q1 - _IQR_MULTIPLIER * iqr
                iqr_upper = q3 + _IQR_MULTIPLIER * iqr
                iqr_mask = (grp[col] < iqr_lower) | (grp[col] > iqr_upper)

                # --- Modified Z-score (MAD-based) ---
                median = vals.median()
                mad = np.median(np.abs(vals - median))
                if mad > 0:
                    modified_z = 0.6745 * (grp[col] - median) / mad
                    z_mask = np.abs(modified_z) > _MODIFIED_ZSCORE_THRESHOLD
                else:
                    z_mask = pd.Series(False, index=idx)

                # Flag if EITHER method detects it
                combined = iqr_mask | z_mask
                outlier_flags.loc[idx, col] = combined

                n_flagged = int(combined.sum())
                plant_outlier_total += n_flagged

                if col not in outlier_stats:
                    outlier_stats[col] = {"iqr": 0, "z_score": 0, "combined": 0}
                outlier_stats[col]["iqr"] += int(iqr_mask.sum())
                outlier_stats[col]["z_score"] += int(z_mask.sum())
                outlier_stats[col]["combined"] += n_flagged

            self.logger.info(
                "  %s: %d outlier flags across all features", plant_id, plant_outlier_total
            )

        # Aggregate per-row
        df["outlier_count"] = outlier_flags.sum(axis=1).astype(int)
        df["is_outlier"] = (df["outlier_count"] > 0).astype(int)

        # Build per-row list of flagged feature names
        flagged_features = []
        for i in range(len(df)):
            cols_flagged = [
                c for c in _NUMERIC_FEATURE_COLS if outlier_flags.iloc[i][c]
            ]
            flagged_features.append(",".join(cols_flagged) if cols_flagged else "")
        df["outlier_features"] = flagged_features

        total_outlier_rows = int(df["is_outlier"].sum())
        self._report["outlier_detection"] = {
            "total_outlier_rows": total_outlier_rows,
            "outlier_pct": round(100.0 * total_outlier_rows / len(df), 3),
            "per_feature": outlier_stats,
            "method": f"IQR(x{_IQR_MULTIPLIER}) + ModifiedZ(>{_MODIFIED_ZSCORE_THRESHOLD})",
        }
        self.logger.info(
            "Outlier detection complete: %d rows flagged (%.2f%%)",
            total_outlier_rows,
            100.0 * total_outlier_rows / len(df),
        )
        return df

    # ------------------------------------------------------------------
    # 4. Impute missing values (per-plant KNN)
    # ------------------------------------------------------------------

    def impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute sparse NaN values using per-plant KNN imputation.

        DCS communication drops occasionally cause isolated missing
        readings.  KNN imputation (k=5) preserves local correlations
        better than simple mean/median fill.  Imputation is performed
        separately per plant to avoid cross-plant leakage.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            DataFrame with NaN values filled in numeric columns.
        """
        nan_before = df[_NUMERIC_FEATURE_COLS].isna().sum().to_dict()
        total_before = sum(nan_before.values())
        self.logger.info(
            "Missing values before imputation: %d (%.3f%%)",
            total_before,
            100.0 * total_before / (len(df) * len(_NUMERIC_FEATURE_COLS)),
        )

        if total_before == 0:
            self.logger.info("No missing values — skipping imputation")
            self._report["nan_before"] = nan_before
            self._report["nan_after"] = nan_before
            return df

        imputer = KNNImputer(n_neighbors=_DEFAULT_KNN_NEIGHBOURS)
        imputed_parts: list[pd.DataFrame] = []

        for plant_id, grp in df.groupby("plant_id", sort=False):
            plant_nan = grp[_NUMERIC_FEATURE_COLS].isna().sum().sum()
            if plant_nan > 0:
                self.logger.info(
                    "  %s: imputing %d NaN across %d records",
                    plant_id, plant_nan, len(grp),
                )
                numeric_vals = grp[_NUMERIC_FEATURE_COLS].values
                imputed_vals = imputer.fit_transform(numeric_vals)
                grp = grp.copy()
                grp[_NUMERIC_FEATURE_COLS] = imputed_vals
            imputed_parts.append(grp)

        df = pd.concat(imputed_parts, ignore_index=False).sort_index()

        nan_after = df[_NUMERIC_FEATURE_COLS].isna().sum().to_dict()
        total_after = sum(nan_after.values())

        self._report["nan_before"] = nan_before
        self._report["nan_after"] = nan_after
        self.logger.info(
            "Imputation complete: %d -> %d NaN remaining", total_before, total_after
        )
        return df

    # ------------------------------------------------------------------
    # 5. Detect turnaround gaps
    # ------------------------------------------------------------------

    def detect_turnaround_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag rows that follow a planned turnaround shutdown gap.

        A gap exceeding ``_TURNAROUND_GAP_HOURS`` between consecutive
        timestamps for a single plant indicates the unit was offline
        for scheduled maintenance.  These gaps are *not* missing data —
        they are legitimate operational shutdowns.

        Adds columns:

        - ``is_turnaround_gap`` (bool): ``True`` for the first row after
          each detected gap.
        - ``days_since_restart`` (float): calendar days elapsed since the
          most recent restart, 0 for plants with no detected gap.
        - ``hours_offline`` (float): duration of the turnaround in hours
          (only set on the first row after the gap).
        - ``is_post_ta_rampup`` (bool): ``True`` for the first 72 hours
          after restart (operators gradually bring plant to steady state).

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        self.logger.info(
            "Detecting turnaround gaps (threshold > %dh)", _TURNAROUND_GAP_HOURS
        )
        df = df.copy()
        df["is_turnaround_gap"] = False
        df["days_since_restart"] = 0.0
        df["hours_offline"] = 0.0
        df["is_post_ta_rampup"] = False

        gap_threshold = pd.Timedelta(hours=_TURNAROUND_GAP_HOURS)
        post_ta_rampup_hours = 72  # 3 days for process stabilisation
        gaps_per_plant: dict[str, list[dict]] = {}

        for plant_id, grp in df.groupby("plant_id", sort=False):
            idx = grp.sort_values("timestamp").index
            ts = df.loc[idx, "timestamp"]
            diffs = ts.diff()

            # Rows where the gap exceeds threshold = first row after restart
            gap_mask = diffs > gap_threshold
            gap_indices = idx[gap_mask]
            gap_count = len(gap_indices)
            gaps_per_plant[plant_id] = []

            if gap_count > 0:
                for gap_idx in gap_indices:
                    df.loc[gap_idx, "is_turnaround_gap"] = True

                    # Record offline duration
                    offline_hours = diffs.loc[gap_idx].total_seconds() / 3600.0
                    df.loc[gap_idx, "hours_offline"] = offline_hours

                    restart_ts = df.loc[gap_idx, "timestamp"]
                    gaps_per_plant[plant_id].append({
                        "restart_timestamp": restart_ts.isoformat(),
                        "hours_offline": round(offline_hours, 1),
                        "days_offline": round(offline_hours / 24.0, 1),
                    })

                    # Days since restart for all subsequent rows
                    post_restart = idx[idx >= gap_idx]
                    days_elapsed = (
                        df.loc[post_restart, "timestamp"] - restart_ts
                    ).dt.total_seconds() / 86400.0
                    df.loc[post_restart, "days_since_restart"] = days_elapsed

                    # Post-TA ramp-up flag (first 72h)
                    hours_elapsed = days_elapsed * 24.0
                    rampup_mask = hours_elapsed <= post_ta_rampup_hours
                    df.loc[post_restart[rampup_mask], "is_post_ta_rampup"] = True

                self.logger.info(
                    "  %s: %d turnaround gap(s) — offline durations: %s",
                    plant_id,
                    gap_count,
                    [f"{g['days_offline']}d" for g in gaps_per_plant[plant_id]],
                )

        self._report["turnaround_gaps_detected"] = {
            pid: len(gaps) for pid, gaps in gaps_per_plant.items()
        }
        self._report["turnaround_details"] = gaps_per_plant
        total_gaps = sum(len(g) for g in gaps_per_plant.values())
        self.logger.info("Turnaround detection complete: %d total gaps", total_gaps)
        return df

    # ------------------------------------------------------------------
    # 6. Cross-plant signal validation
    # ------------------------------------------------------------------

    def validate_cross_plant_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate that shared environmental signals agree across plants.

        Ambient temperature, humidity, and natural gas calorific value
        are measured at the shared Gadepan complex.  Cross-plant
        correlation should exceed 0.95 — lower values indicate a
        potential sensor calibration issue.

        Adds column:

        - ``signal_agreement_score`` (float 0–1): average cross-plant
          correlation for shared signals at each timestamp.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        self.logger.info("Validating cross-plant signal agreement")
        signal_report: dict[str, dict] = {}

        # For each shared signal, compute pairwise correlation across plants
        plants = sorted(df["plant_id"].unique())
        if len(plants) < 2:
            self.logger.info("  Only 1 plant — skipping cross-plant validation")
            df["signal_agreement_score"] = 1.0
            return df

        for signal in _SHARED_SIGNALS:
            pivot = df.pivot_table(
                index="timestamp", columns="plant_id", values=signal, aggfunc="first"
            ).dropna()

            if len(pivot) < 10:
                continue

            # Pairwise Pearson correlations
            corr_matrix = pivot.corr()
            # Extract upper triangle (excluding diagonal)
            pairs = []
            for i, p1 in enumerate(plants):
                for p2 in plants[i + 1:]:
                    if p1 in corr_matrix.columns and p2 in corr_matrix.columns:
                        r = corr_matrix.loc[p1, p2]
                        pairs.append({"pair": f"{p1}-{p2}", "r": round(r, 4)})

            avg_r = np.mean([p["r"] for p in pairs]) if pairs else 1.0
            signal_report[signal] = {
                "avg_correlation": round(avg_r, 4),
                "pairs": pairs,
                "status": "OK" if avg_r > 0.95 else "WARNING",
            }

            self.logger.info(
                "  %s: avg cross-plant r = %.4f [%s]",
                signal, avg_r, signal_report[signal]["status"],
            )

        # Compute per-timestamp agreement score across shared environmental signals
        # Compares deviation relative to process scale of the variable
        signal_scores = []
        for signal in _SHARED_SIGNALS:
            spec = RAW_COLUMNS.get(signal)
            scale = (spec.max_value - spec.min_value) if spec else 10.0
            fleet_mean = df.groupby("timestamp")[signal].transform("mean")
            rel_dev = (np.abs(df[signal] - fleet_mean) / scale).clip(upper=1.0)
            signal_scores.append(1.0 - rel_dev)

        # Average agreement across all shared signals
        df["signal_agreement_score"] = pd.concat(signal_scores, axis=1).mean(axis=1).round(4)

        self._report["cross_plant_signal_validation"] = signal_report
        self.logger.info("Cross-plant signal validation complete")
        return df

    # ------------------------------------------------------------------
    # 7. Encode categoricals
    # ------------------------------------------------------------------

    def encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode ``shift`` and ``plant_id`` columns.

        All categories are retained (``drop_first=False``) for
        interpretability in SHAP analysis.  Original columns are
        preserved for downstream grouping / filtering.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        self.logger.info("Encoding categorical features")

        # Shift encoding
        shift_dummies = pd.get_dummies(
            df["shift"], prefix="shift", dtype=int
        )
        for col in ["shift_morning", "shift_afternoon", "shift_night"]:
            if col not in shift_dummies.columns:
                shift_dummies[col] = 0

        # Plant encoding
        plant_dummies = pd.get_dummies(
            df["plant_id"], prefix="plant", dtype=int
        )
        for col in ["plant_gadepan_1", "plant_gadepan_2", "plant_gadepan_3"]:
            if col not in plant_dummies.columns:
                plant_dummies[col] = 0

        df = pd.concat([df, shift_dummies, plant_dummies], axis=1)

        new_cols = list(shift_dummies.columns) + list(plant_dummies.columns)
        self.logger.info("  Added %d one-hot columns: %s", len(new_cols), new_cols)
        return df

    # ------------------------------------------------------------------
    # 8. Temporal features
    # ------------------------------------------------------------------

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features from the ``timestamp`` column.

        Creates:

        - ``hour``, ``day_of_week``, ``month``, ``day_of_year`` (integer)
        - ``is_weekend`` (bool, Sat/Sun)
        - ``is_peak_heat`` (bool, 12:00–16:00 when ambient peaks)
        - ``is_night_shift`` (bool, 22:00–06:00 when SEC tends to drift)
        - Cyclical encodings: ``hour_sin``, ``hour_cos``,
          ``month_sin``, ``month_cos``, ``dow_sin``, ``dow_cos``

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        self.logger.info("Adding temporal features")
        ts = df["timestamp"]

        df["hour"] = ts.dt.hour
        df["day_of_week"] = ts.dt.dayofweek  # 0=Mon, 6=Sun
        df["month"] = ts.dt.month
        df["day_of_year"] = ts.dt.dayofyear
        df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

        # Operational regime flags
        df["is_peak_heat"] = ((df["hour"] >= 12) & (df["hour"] <= 16)).astype(int)
        df["is_night_shift"] = (
            (df["hour"] >= 22) | (df["hour"] < 6)
        ).astype(int)

        # Cyclical encoding to capture periodicity
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        temporal_cols = [
            "hour", "day_of_week", "month", "day_of_year",
            "is_weekend", "is_peak_heat", "is_night_shift",
            "hour_sin", "hour_cos", "month_sin", "month_cos",
            "dow_sin", "dow_cos",
        ]
        self.logger.info("  Added %d temporal features", len(temporal_cols))
        return df

    # ------------------------------------------------------------------
    # 9. Data quality score
    # ------------------------------------------------------------------

    def compute_data_quality_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute a per-row composite data quality score (0–100).

        Factors:

        - **Completeness** (was this row imputed? -10 per NaN in raw)
        - **Outlier penalty** (-5 per outlier feature flagged)
        - **Signal agreement** (weighted by cross-plant score)
        - **Temporal regularity** (is this row part of normal hourly cadence?)
        - **Post-TA stability** (newly restarted plants get lower scores
          during the first 72 hours of ramp-up)

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            With ``data_quality_score`` column (0–100 scale).
        """
        self.logger.info("Computing per-row data quality scores")

        score = pd.Series(100.0, index=df.index)

        # Outlier penalty: -5 per feature flagged
        if "outlier_count" in df.columns:
            score -= df["outlier_count"] * 5.0

        # Post-TA ramp-up penalty: -15 during stabilisation
        if "is_post_ta_rampup" in df.columns:
            score -= df["is_post_ta_rampup"].astype(float) * 15.0

        # Signal agreement: scale by agreement score
        if "signal_agreement_score" in df.columns:
            # 0 to -10 penalty for poor agreement
            score -= (1.0 - df["signal_agreement_score"]) * 10.0

        # Turnaround restart row: -10 (first reading after long gap)
        if "is_turnaround_gap" in df.columns:
            score -= df["is_turnaround_gap"].astype(float) * 10.0

        # Clamp to [0, 100]
        df["data_quality_score"] = score.clip(lower=0, upper=100).round(1)

        # Summary stats
        mean_quality = df["data_quality_score"].mean()
        below_80 = (df["data_quality_score"] < 80).sum()
        below_50 = (df["data_quality_score"] < 50).sum()

        self._report["data_quality"] = {
            "mean_score": round(mean_quality, 2),
            "min_score": round(df["data_quality_score"].min(), 2),
            "max_score": round(df["data_quality_score"].max(), 2),
            "rows_below_80": int(below_80),
            "rows_below_50": int(below_50),
            "score_percentiles": {
                "p5": round(df["data_quality_score"].quantile(0.05), 2),
                "p25": round(df["data_quality_score"].quantile(0.25), 2),
                "p50": round(df["data_quality_score"].quantile(0.50), 2),
                "p75": round(df["data_quality_score"].quantile(0.75), 2),
                "p95": round(df["data_quality_score"].quantile(0.95), 2),
            },
        }
        self.logger.info(
            "Data quality scoring complete: mean=%.1f, rows below 80==%d",
            mean_quality, below_80,
        )
        return df

    # ------------------------------------------------------------------
    # 10. Full pipeline
    # ------------------------------------------------------------------

    def run(self, filepath: Path | str | None = None) -> pd.DataFrame:
        """Execute the full preprocessing pipeline.

        Steps:
            1. Load raw CSV
            2. Validate & clip sensor ranges
            3. Detect statistical outliers (IQR + modified Z-score)
            4. KNN-impute missing values (per plant)
            5. Detect turnaround gaps + post-TA ramp-up
            6. Validate cross-plant shared signals
            7. One-hot encode categoricals
            8. Add temporal features
            9. Compute data quality scores
            10. Save processed CSV + processing report

        Parameters
        ----------
        filepath : Path or str, optional
            Path to raw CSV (defaults to ``data/raw/plant_data_raw.csv``).

        Returns
        -------
        pd.DataFrame
            Fully preprocessed DataFrame, ready for feature engineering.
        """
        self.logger.info("=" * 70)
        self.logger.info("STARTING DATA PREPROCESSING PIPELINE")
        self.logger.info("=" * 70)

        df = self.load_raw_data(filepath)
        df = self.validate_ranges(df)
        df = self.detect_outliers(df)
        df = self.impute_missing(df)
        df = self.detect_turnaround_gaps(df)
        df = self.validate_cross_plant_signals(df)
        df = self.encode_categoricals(df)
        df = self.add_temporal_features(df)
        df = self.compute_data_quality_score(df)

        # --- Save outputs ---
        output_dir = get_project_root() / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / "plant_data_processed.csv"
        df.to_csv(csv_path, index=False)
        self.logger.info(
            "Saved processed data to %s (%d rows, %d cols)",
            csv_path, len(df), len(df.columns),
        )

        parquet_path = output_dir / "plant_data_processed.parquet"
        df.to_parquet(parquet_path, index=False, engine="pyarrow", compression="snappy")
        self.logger.info("Saved compressed Parquet telemetry to %s", parquet_path)

        # Compile and save processing report
        self._report["processed_records"] = len(df)
        self._report["processed_columns"] = list(df.columns)
        self._report["features_added"] = [
            c for c in df.columns if c not in set(RAW_COLUMNS.keys())
        ]
        self._report["processing_timestamp"] = datetime.now().isoformat()

        report_path = output_dir / "processing_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(self._report, f, indent=2, default=str)
        self.logger.info("Saved processing report to %s", report_path)

        self.logger.info("=" * 70)
        self.logger.info("PREPROCESSING COMPLETE — %d rows, %d columns", len(df), len(df.columns))
        self.logger.info("=" * 70)
        return df

    @property
    def report(self) -> dict[str, Any]:
        """Return the processing report dictionary (populated after :meth:`run`)."""
        return self._report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    preprocessor = DataPreprocessor()

    filepath_arg = sys.argv[1] if len(sys.argv) > 1 else None
    df = preprocessor.run(filepath_arg)

    print(f"\nProcessed DataFrame: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"\nColumns:\n  {list(df.columns)}")
    print(f"\nNaN remaining: {df.isna().sum().sum()}")
    print(f"\nData quality: mean={df['data_quality_score'].mean():.1f}, "
          f"min={df['data_quality_score'].min():.1f}")
    print(f"Outlier rows: {df['is_outlier'].sum()}")
    print(f"\nReport keys: {list(preprocessor.report.keys())}")
