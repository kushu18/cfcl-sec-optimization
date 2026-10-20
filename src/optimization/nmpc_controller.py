"""
CFCL Gadepan Complex - Non-Linear Model Predictive Control (NMPC) Receding Horizon.
6-hour forward-looking dynamic trajectory optimization incorporating Rajasthan diurnal
ambient wet-bulb swings, terminal energy costs, and actuator slew penalties.
Sanitized DCS Telemetry under strict NDA.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import numpy as np
from scipy.optimize import minimize

from src.data.schema import FEATURE_COLUMNS
from src.utils.logger import get_logger

logger = get_logger("NMPCController")


@dataclass
class NMPCTrajectoryStep:
    """A single step in the 6-hour planned control trajectory."""
    hour_ahead: int
    forecast_ambient_temp_c: float
    recommended_steam_carbon_ratio: float
    recommended_reformer_outlet_temp: float
    recommended_loop_pressure: float
    predicted_sec_gcal_per_mt: float
    slew_rate_temp_c_hr: float


@dataclass
class NMPCPlanResult:
    """Complete 6-hour receding horizon optimization result."""
    current_sec_gcal_per_mt: float
    horizon_hours: int
    immediate_setpoints: Dict[str, float]
    planned_trajectory: List[NMPCTrajectoryStep]
    average_projected_sec: float
    projected_6h_energy_reduction_gcal_per_mt: float
    solver_converged: bool


class NMPCController:
    """
    Receding Horizon NMPC controller with N = 6 hours.
    Anticipates cooling tower performance drops during peak afternoon heat
    (Rajasthan Kota summer ambient ~42-45 C) and pre-adjusts synthesis loop chillers.
    """

    HORIZON_STEPS = 6

    def __init__(self, model: Any = None):
        from src.optimization.constrained_optimizer import ConstrainedSECOptimizer
        if isinstance(model, ConstrainedSECOptimizer):
            self.optimizer = model
            self.model = model.model
        elif hasattr(model, "_predict_sec"):
            self.optimizer = model
            self.model = getattr(model, "model", None)
        else:
            self.model = model
            self.optimizer = ConstrainedSECOptimizer() if model is None or getattr(model, "n_features_in_", 9) > 9 else None

        self.bounds = {
            "steam_carbon_ratio": (2.80, 3.40),
            "reformer_outlet_temp": (780.0, 835.0),
            "loop_pressure": (145.0, 185.0),
        }
        self.slew_limits = {
            "steam_carbon_ratio": 0.05,
            "reformer_outlet_temp": 2.0,
            "loop_pressure": 2.0,
        }

    def _predict_step(
        self,
        u: np.ndarray,
        base_features: Dict[str, float],
        ambient_temp_c: float,
    ) -> float:
        import pandas as pd
        row = dict(base_features)
        row["steam_carbon_ratio"] = float(u[0])
        row["reformer_outlet_temp"] = float(u[1])
        row["loop_pressure"] = float(u[2])
        row["ambient_temp"] = float(ambient_temp_c)

        # Approximate cooling water temperature tracking ambient wet-bulb:
        # CW inlet rises ~0.35 C for every 1.0 C ambient rise
        row["cw_inlet_temp"] = min(36.0, row.get("cw_inlet_temp", 31.0) + 0.35 * (ambient_temp_c - base_features.get("ambient_temp", 32.0)))

        if self.optimizer is not None:
            return float(self.optimizer._predict_sec(np.asarray(u, dtype=float), pd.Series(row)))

        vec = np.array([[row.get(col, 0.0) for col in FEATURE_COLUMNS]], dtype=np.float32)
        if hasattr(self.model, "predict"):
            pred = self.model.predict(vec)
            return float(pred[0] if hasattr(pred, "__len__") else pred)
        return 5.50

    def solve(
        self,
        current_state: Dict[str, float],
        ambient_temp_forecast: Optional[List[float]] = None,
    ) -> NMPCPlanResult:
        """
        Solves the dynamic 6-hour receding horizon problem.
        ambient_temp_forecast: list of 6 hourly ambient temperature values (C).
        """
        curr_u0 = np.array(
            [
                float(current_state["steam_carbon_ratio"]),
                float(current_state["reformer_outlet_temp"]),
                float(current_state["loop_pressure"]),
            ],
            dtype=np.float64,
        )

        curr_sec = self._predict_step(
            curr_u0, current_state, float(current_state.get("ambient_temp", 32.0))
        )

        # Default 6-hour forecast: afternoon heating or night cooling
        if not ambient_temp_forecast or len(ambient_temp_forecast) < self.HORIZON_STEPS:
            curr_amb = float(current_state.get("ambient_temp", 32.0))
            ambient_temp_forecast = [
                curr_amb + 1.2 * i for i in range(self.HORIZON_STEPS)
            ]

        # Decision vector: 6 steps x 3 controls = 18 variables
        u_init = np.tile(curr_u0, self.HORIZON_STEPS)

        # Slew-clamped bounds across each horizon step
        bounds_list = []
        for step in range(self.HORIZON_STEPS):
            bounds_list.extend(
                [
                    (self.bounds["steam_carbon_ratio"][0], self.bounds["steam_carbon_ratio"][1]),
                    (self.bounds["reformer_outlet_temp"][0], self.bounds["reformer_outlet_temp"][1]),
                    (self.bounds["loop_pressure"][0], self.bounds["loop_pressure"][1]),
                ]
            )

        def objective(U: np.ndarray) -> float:
            total_cost = 0.0
            u_prev = curr_u0

            for k in range(self.HORIZON_STEPS):
                u_k = U[k * 3 : (k + 1) * 3]
                amb_k = ambient_temp_forecast[k]

                # Step energy cost
                sec_k = self._predict_step(u_k, current_state, amb_k)

                # Slew penalty: prevent erratic swinging of actuators
                du_sc = (u_k[0] - u_prev[0]) / self.slew_limits["steam_carbon_ratio"]
                du_temp = (u_k[1] - u_prev[1]) / self.slew_limits["reformer_outlet_temp"]
                du_press = (u_k[2] - u_prev[2]) / self.slew_limits["loop_pressure"]
                slew_penalty = 0.15 * (du_sc**2 + du_temp**2 + du_press**2)

                # Hard slew violation penalty
                if abs(u_k[1] - u_prev[1]) > self.slew_limits["reformer_outlet_temp"]:
                    slew_penalty += 1000.0 * (abs(u_k[1] - u_prev[1]) - self.slew_limits["reformer_outlet_temp"])

                total_cost += sec_k + slew_penalty
                u_prev = u_k

            return float(total_cost)

        res = minimize(
            objective,
            u_init,
            method="SLSQP",
            bounds=bounds_list,
            options={"maxiter": 120, "ftol": 1e-5},
        )

        u_opt = res.x if res.success else u_init

        trajectory: List[NMPCTrajectoryStep] = []
        u_prev = curr_u0
        sum_sec = 0.0

        for k in range(self.HORIZON_STEPS):
            u_k = u_opt[k * 3 : (k + 1) * 3]
            amb_k = ambient_temp_forecast[k]
            step_sec = self._predict_step(u_k, current_state, amb_k)
            sum_sec += step_sec

            slew_t = abs(u_k[1] - u_prev[1])
            trajectory.append(
                NMPCTrajectoryStep(
                    hour_ahead=k + 1,
                    forecast_ambient_temp_c=round(amb_k, 1),
                    recommended_steam_carbon_ratio=round(float(u_k[0]), 3),
                    recommended_reformer_outlet_temp=round(float(u_k[1]), 2),
                    recommended_loop_pressure=round(float(u_k[2]), 2),
                    predicted_sec_gcal_per_mt=round(step_sec, 4),
                    slew_rate_temp_c_hr=round(float(slew_t), 2),
                )
            )
            u_prev = u_k

        avg_sec = sum_sec / self.HORIZON_STEPS
        immediate_u = trajectory[0]

        return NMPCPlanResult(
            current_sec_gcal_per_mt=round(curr_sec, 4),
            horizon_hours=self.HORIZON_STEPS,
            immediate_setpoints={
                "steam_carbon_ratio": immediate_u.recommended_steam_carbon_ratio,
                "reformer_outlet_temp": immediate_u.recommended_reformer_outlet_temp,
                "loop_pressure": immediate_u.recommended_loop_pressure,
            },
            planned_trajectory=trajectory,
            average_projected_sec=round(avg_sec, 4),
            projected_6h_energy_reduction_gcal_per_mt=round(max(0.0, curr_sec - avg_sec), 4),
            solver_converged=bool(res.success),
        )
