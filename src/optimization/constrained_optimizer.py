"""Constrained Industrial Optimizer for CFCL Specific Energy Consumption (SEC).

Solves the real-time steady-state optimization problem:
    Minimize: SEC(x_mv, x_dv) via trained ML surrogate (Random Forest / XGBoost)
    Subject to:
        1. Hard Plant Safety Bounds (DCS Governor Limits):
           - S/C ratio: [2.80, 3.40]
           - Reformer Outlet Temp: [780.0, 835.0] °C
           - Synthesis Loop Pressure: [145.0, 185.0] kg/cm²
        2. Hourly Slew-Rate Constraints (DCS Ramp-Rate Interlocks):
           - |Δ(S/C)| <= 0.05 / hour
           - |ΔT_reformer| <= 2.0 °C / hour (thermal fatigue prevention on Incoloy 800H tubes)
           - |ΔP_loop| <= 2.0 kg/cm² / hour (syngas compressor surge margin)
        3. Dynamic Physical Feature Reconstruction:
           - Recalculates all dependent interaction terms and optimal setpoint deviations
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.data.schema import CONTROLLABLE_FEATURES, TARGET_COLUMN
from src.utils.config_loader import get_paths, get_project_root, load_config
from src.utils.logger import get_logger


@dataclass
class OptimizationResult:
    """Structured result for a single operational optimization timestep."""
    timestamp: str
    plant_id: str
    baseline_sec: float
    optimized_sec: float
    delta_sec: float
    percent_reduction: float
    current_setpoints: Dict[str, float]
    recommended_setpoints: Dict[str, float]
    setpoint_adjustments: Dict[str, float]
    active_constraints: List[str]
    success: bool
    iterations: int


class ConstrainedSECOptimizer:
    """Industrial SLSQP Optimizer with DCS safety bounds and slew-rate interlocks."""

    # Default Hard Safety Bounds (from plant_config.yaml)
    SAFETY_BOUNDS: Dict[str, Tuple[float, float]] = {
        "steam_carbon_ratio": (2.80, 3.40),
        "reformer_outlet_temp": (780.0, 835.0),
        "loop_pressure": (145.0, 185.0),
    }

    # Maximum Allowable Rate of Change (per 1-hour control interval)
    SLEW_RATE_LIMITS: Dict[str, float] = {
        "steam_carbon_ratio": 0.05,        # max 0.05 ratio change per hour
        "reformer_outlet_temp": 2.0,        # max 2.0 °C change per hour (thermal stress limit)
        "loop_pressure": 2.0,               # max 2.0 kg/cm² change per hour (compressor surge margin)
    }

    OPTIMAL_MIDPOINTS: Dict[str, float] = {
        "steam_carbon_ratio": 3.10,        # midpoint of [3.0, 3.2]
        "reformer_outlet_temp": 807.5,      # midpoint of [800.0, 815.0]
        "loop_pressure": 162.5,             # midpoint of [155.0, 170.0]
    }

    def __init__(self, model_artifact_path: Optional[Path | str] = None, logger: Optional[Any] = None) -> None:
        self.logger = logger or get_logger("constrained_optimizer")
        self.paths = get_paths()
        self.model_cfg = load_config("model")
        self.opt_cfg = self.model_cfg.get("optimizer", {})

        # Load Champion Model (Random Forest or XGBoost)
        models_dir = self.paths["models"]
        if model_artifact_path is None:
            # Check model leaderboard for champion
            lb_file = models_dir / "model_leaderboard.json"
            if lb_file.exists():
                with open(lb_file, "r", encoding="utf-8") as f:
                    lb = json.load(f)
                champ = lb.get("champion_model", "random_forest").lower()
                if "random" in champ:
                    model_artifact_path = models_dir / "random_forest.joblib"
                elif "xgboost" in champ:
                    model_artifact_path = models_dir / "xgboost_sec.joblib"
                else:
                    model_artifact_path = models_dir / "random_forest.joblib"
            else:
                model_artifact_path = models_dir / "random_forest.joblib"

        model_artifact_path = Path(model_artifact_path)
        if not model_artifact_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {model_artifact_path}")

        artifact = joblib.load(model_artifact_path)
        self.model = artifact["model"]
        self.feature_names: List[str] = artifact["feature_names"]
        self.logger.info("Loaded surrogate model from %s with %d features.", model_artifact_path.name, len(self.feature_names))

        # Check if linear scaler exists (e.g. if linear baseline was loaded)
        self.scaler = artifact.get("scaler", None)

    def _reconstruct_features(self, x_mv: np.ndarray, base_row: pd.Series) -> pd.DataFrame:
        """Dynamically reconstruct dependent physical interactions and deviation features.
        
        Parameters
        ----------
        x_mv : np.ndarray
            Current candidate vector [steam_carbon_ratio, reformer_outlet_temp, loop_pressure].
        base_row : pd.Series
            Original row containing uncontrollable disturbance variables and metadata.
            
        Returns
        -------
        pd.DataFrame
            Single-row DataFrame matching self.feature_names.
        """
        sc_val, ref_val, press_val = float(x_mv[0]), float(x_mv[1]), float(x_mv[2])

        # Start with copy of base row
        row_dict = base_row.to_dict()

        # Update core manipulated variables
        row_dict["steam_carbon_ratio"] = sc_val
        row_dict["reformer_outlet_temp"] = ref_val
        row_dict["loop_pressure"] = press_val

        # 1. Dependent Thermodynamic Interactions
        row_dict["steam_reformer_interaction"] = sc_val * ref_val
        row_dict["pressure_temp_ratio"] = press_val / ref_val if ref_val > 0 else 0.0
        row_dict["steam_pressure_interaction"] = sc_val * press_val

        # 2. Optimal Setpoint Deviations
        row_dict["steam_carbon_ratio_deviation"] = sc_val - self.OPTIMAL_MIDPOINTS["steam_carbon_ratio"]
        row_dict["reformer_outlet_temp_deviation"] = ref_val - self.OPTIMAL_MIDPOINTS["reformer_outlet_temp"]
        row_dict["loop_pressure_deviation"] = press_val - self.OPTIMAL_MIDPOINTS["loop_pressure"]

        # 3. Binary Outside Optimal Flags
        row_dict["steam_carbon_ratio_outside_optimal"] = 1.0 if (sc_val < 3.0 or sc_val > 3.2) else 0.0
        row_dict["reformer_outlet_temp_outside_optimal"] = 1.0 if (ref_val < 800.0 or ref_val > 815.0) else 0.0
        row_dict["loop_pressure_outside_optimal"] = 1.0 if (press_val < 155.0 or press_val > 170.0) else 0.0

        # 4. Thermal load index
        cw_temp = row_dict.get("cw_inlet_temp", 30.0)
        amb_temp = row_dict.get("ambient_temp", 32.0)
        row_dict["thermal_load_index"] = (amb_temp - cw_temp) / ref_val if ref_val > 0 else 0.0

        # Build feature vector aligned with model feature_names
        feat_vals = [row_dict.get(col, 0.0) for col in self.feature_names]
        return pd.DataFrame([feat_vals], columns=self.feature_names).fillna(0)

    def extract_features(self, row_dict: Dict[str, Any]) -> pd.DataFrame:
        """Extract full model-aligned feature DataFrame from a telemetry dictionary."""
        base_series = pd.Series(row_dict)
        x0 = np.array([
            float(row_dict.get("steam_carbon_ratio", 3.05)),
            float(row_dict.get("reformer_outlet_temp", 805.0)),
            float(row_dict.get("loop_pressure", 160.0)),
        ])
        return self._reconstruct_features(x0, base_series)

    def _predict_sec(self, x_mv: np.ndarray, base_row: pd.Series) -> float:
        """Evaluate surrogate model SEC objective function."""
        X_eval = self._reconstruct_features(x_mv, base_row)
        if self.scaler is not None:
            pred = self.model.predict(self.scaler.transform(X_eval))[0]
        else:
            pred = self.model.predict(X_eval)[0]
        return float(pred)

    def optimize_hour(self, current_row: pd.Series, respect_slew_rate: bool = True) -> OptimizationResult:
        """Optimize operating setpoints for a single DCS hourly snapshot.
        
        Parameters
        ----------
        current_row : pd.Series
            DCS observation containing current setpoints, disturbance variables, and plant ID.
        respect_slew_rate : bool
            Whether to enforce dynamic 1-hour ramp rate limits around current operating point.
            
        Returns
        -------
        OptimizationResult
            Comprehensive optimization trajectory and recommended setpoint adjustments.
        """
        # Current measured operating setpoints
        x0_sc = float(current_row["steam_carbon_ratio"])
        x0_ref = float(current_row["reformer_outlet_temp"])
        x0_press = float(current_row["loop_pressure"])
        x0 = np.array([x0_sc, x0_ref, x0_press])

        # Current baseline SEC evaluated by surrogate model at current operating point x0
        pred_baseline_sec = self._predict_sec(x0, current_row)
        measured_sec = float(current_row.get(TARGET_COLUMN, np.nan))
        baseline_sec = pred_baseline_sec

        # Determine Effective Optimization Bounds
        bounds: List[Tuple[float, float]] = []
        active_constraints: List[str] = []

        mvs = ["steam_carbon_ratio", "reformer_outlet_temp", "loop_pressure"]
        for i, mv in enumerate(mvs):
            safe_min, safe_max = self.SAFETY_BOUNDS[mv]
            if respect_slew_rate:
                slew = self.SLEW_RATE_LIMITS[mv]
                eff_min = max(safe_min, x0[i] - slew)
                eff_max = min(safe_max, x0[i] + slew)
            else:
                eff_min, eff_max = safe_min, safe_max

            bounds.append((eff_min, eff_max))

        # Objective function wrapper
        def objective(x: np.ndarray) -> float:
            return self._predict_sec(x, current_row)

        # Multi-start SLSQP search (ensures escaping local flat regions)
        starts = [x0]
        for _ in range(4):
            rand_start = np.array([
                np.random.uniform(bounds[0][0], bounds[0][1]),
                np.random.uniform(bounds[1][0], bounds[1][1]),
                np.random.uniform(bounds[2][0], bounds[2][1]),
            ])
            starts.append(rand_start)

        best_opt_x = x0
        best_opt_sec = pred_baseline_sec
        best_success = False
        total_iters = 0

        for start_vec in starts:
            res = minimize(
                fun=objective,
                x0=start_vec,
                method="SLSQP",
                bounds=bounds,
                options={"ftol": 1e-6, "maxiter": 150, "disp": False},
            )
            total_iters += res.nit
            if res.success and res.fun < best_opt_sec:
                best_opt_sec = float(res.fun)
                best_opt_x = res.x
                best_success = True

        # Check which bounds were reached (active constraints)
        for i, mv in enumerate(mvs):
            if np.isclose(best_opt_x[i], bounds[i][0], atol=1e-3):
                active_constraints.append(f"{mv}_lower_bound")
            elif np.isclose(best_opt_x[i], bounds[i][1], atol=1e-3):
                active_constraints.append(f"{mv}_upper_bound")

        delta_sec = baseline_sec - best_opt_sec
        pct_reduction = (delta_sec / baseline_sec) * 100.0 if baseline_sec > 0 else 0.0

        # Construct recommended setpoints dict
        curr_setpoints = {"steam_carbon_ratio": x0_sc, "reformer_outlet_temp": x0_ref, "loop_pressure": x0_press}
        rec_setpoints = {
            "steam_carbon_ratio": round(float(best_opt_x[0]), 3),
            "reformer_outlet_temp": round(float(best_opt_x[1]), 2),
            "loop_pressure": round(float(best_opt_x[2]), 2),
        }
        adjustments = {
            "steam_carbon_ratio": round(float(best_opt_x[0] - x0_sc), 3),
            "reformer_outlet_temp": round(float(best_opt_x[1] - x0_ref), 2),
            "loop_pressure": round(float(best_opt_x[2] - x0_press), 2),
        }

        ts_str = str(current_row.get("timestamp", "unknown"))
        plant_id = str(current_row.get("plant_id", "fleet"))

        return OptimizationResult(
            timestamp=ts_str,
            plant_id=plant_id,
            baseline_sec=round(baseline_sec, 4),
            optimized_sec=round(best_opt_sec, 4),
            delta_sec=round(delta_sec, 4),
            percent_reduction=round(pct_reduction, 3),
            current_setpoints=curr_setpoints,
            recommended_setpoints=rec_setpoints,
            setpoint_adjustments=adjustments,
            active_constraints=active_constraints,
            success=best_success,
            iterations=total_iters,
        )

    def optimize_batch(self, df: pd.DataFrame, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Run batch optimization over operational historical hours."""
        self.logger.info("Starting batch constrained optimization across %d records...", len(df))
        
        target_df = df.copy()
        if sample_size is not None and len(df) > sample_size:
            target_df = df.sample(n=sample_size, random_state=42).sort_values("timestamp").reset_index(drop=True)

        results = []
        for idx in range(len(target_df)):
            row = target_df.iloc[idx]
            opt_res = self.optimize_hour(row, respect_slew_rate=True)
            results.append({
                "timestamp": opt_res.timestamp,
                "plant_id": opt_res.plant_id,
                "baseline_sec": opt_res.baseline_sec,
                "optimized_sec": opt_res.optimized_sec,
                "delta_sec": opt_res.delta_sec,
                "percent_reduction": opt_res.percent_reduction,
                "cur_sc": opt_res.current_setpoints["steam_carbon_ratio"],
                "opt_sc": opt_res.recommended_setpoints["steam_carbon_ratio"],
                "delta_sc": opt_res.setpoint_adjustments["steam_carbon_ratio"],
                "cur_reformer_temp": opt_res.current_setpoints["reformer_outlet_temp"],
                "opt_reformer_temp": opt_res.recommended_setpoints["reformer_outlet_temp"],
                "delta_reformer_temp": opt_res.setpoint_adjustments["reformer_outlet_temp"],
                "cur_loop_pressure": opt_res.current_setpoints["loop_pressure"],
                "opt_loop_pressure": opt_res.recommended_setpoints["loop_pressure"],
                "delta_loop_pressure": opt_res.setpoint_adjustments["loop_pressure"],
                "active_constraints": ",".join(opt_res.active_constraints),
            })

        res_df = pd.DataFrame(results)
        self.logger.info("Batch optimization complete: Mean SEC reduction = %.4f Gcal/MT (%.2f%%)",
                         res_df["delta_sec"].mean(), res_df["percent_reduction"].mean())
        return res_df


def main() -> None:
    """Run CLI test on holdout data."""
    paths = get_paths()
    df = pd.read_parquet(paths["processed_data"] / "plant_features.parquet").tail(50)
    optimizer = ConstrainedSECOptimizer()
    res_df = optimizer.optimize_batch(df, sample_size=10)
    print(res_df[["timestamp", "plant_id", "baseline_sec", "optimized_sec", "delta_sec"]].head())


if __name__ == "__main__":
    main()
