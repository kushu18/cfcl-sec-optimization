"""
CFCL Gadepan Complex - Counterfactual Recourse & Decision-Support Engine.
Computes minimum-effort actionable setpoint adjustments (Wachter et al. formulation)
to hit target energy savings without exceeding DCS governor slew limits.
Sanitized DCS Telemetry under strict NDA.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import numpy as np
from scipy.optimize import minimize

from src.data.schema import CONTROLLABLE_FEATURES, FEATURE_COLUMNS
from src.utils.logger import get_logger

logger = get_logger("CounterfactualRecourse")


@dataclass
class RecourseAction:
    """Specific actuator adjustment recommendation for control room operators."""
    feature: str
    current_value: float
    recommended_target: float
    adjustment_delta: float
    units: str
    effort_rank: int


@dataclass
class CounterfactualResult:
    """Complete minimum-effort counterfactual recourse recommendation."""
    current_sec_gcal_per_mt: float
    target_sec_gcal_per_mt: float
    achievable_sec_gcal_per_mt: float
    energy_reduction_gcal_per_mt: float
    estimated_daily_savings_inr: float
    total_recourse_effort_score: float
    action_plan: List[RecourseAction]
    operator_executive_summary: str


class CounterfactualExplainer:
    """
    Solves for the minimal distance perturbation to controllable variables:
      u* = argmin_u sum_j w_j * ((u_j - u0_j) / scale_j)^2 + penalty * (SEC(u) - target)^2
    """

    FEATURE_UNITS = {
        "steam_carbon_ratio": "ratio",
        "reformer_outlet_temp": "deg C",
        "loop_pressure": "bar",
    }

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
        self.scales = {
            "steam_carbon_ratio": 0.05,
            "reformer_outlet_temp": 2.0,
            "loop_pressure": 2.0,
        }

    def _predict(self, u: np.ndarray, row: Dict[str, float]) -> float:
        import pandas as pd
        if self.optimizer is not None:
            return float(self.optimizer._predict_sec(np.asarray(u, dtype=float), pd.Series(row)))

        r = dict(row)
        r["steam_carbon_ratio"] = float(u[0])
        r["reformer_outlet_temp"] = float(u[1])
        r["loop_pressure"] = float(u[2])
        vec = np.array([[r.get(col, 0.0) for col in FEATURE_COLUMNS]], dtype=np.float32)
        if hasattr(self.model, "predict"):
            pred = self.model.predict(vec)
            return float(pred[0] if hasattr(pred, "__len__") else pred)
        return 5.50

    def compute_recourse(
        self,
        current_state: Dict[str, float],
        target_reduction_gcal_per_mt: float = 0.030,
        daily_urea_production_mt: float = 1750.0,
    ) -> CounterfactualResult:
        """
        Computes the minimum-effort setpoint trajectory to achieve target reduction.
        """
        u0 = np.array(
            [
                float(current_state["steam_carbon_ratio"]),
                float(current_state["reformer_outlet_temp"]),
                float(current_state["loop_pressure"]),
            ],
            dtype=np.float64,
        )

        curr_sec = self._predict(u0, current_state)
        target_sec = curr_sec - target_reduction_gcal_per_mt

        # Bounds constrained by 1-hour slew rates
        opt_bounds = [
            (
                max(self.bounds["steam_carbon_ratio"][0], u0[0] - self.scales["steam_carbon_ratio"]),
                min(self.bounds["steam_carbon_ratio"][1], u0[0] + self.scales["steam_carbon_ratio"]),
            ),
            (
                max(self.bounds["reformer_outlet_temp"][0], u0[1] - self.scales["reformer_outlet_temp"]),
                min(self.bounds["reformer_outlet_temp"][1], u0[1] + self.scales["reformer_outlet_temp"]),
            ),
            (
                max(self.bounds["loop_pressure"][0], u0[2] - self.scales["loop_pressure"]),
                min(self.bounds["loop_pressure"][1], u0[2] + self.scales["loop_pressure"]),
            ),
        ]

        # Weights: adjust temperature and pressure easily, minimize S/C steam valve swing
        weights = np.array([1.5, 1.0, 1.0])
        scales = np.array([self.scales["steam_carbon_ratio"], self.scales["reformer_outlet_temp"], self.scales["loop_pressure"]])

        def loss(u: np.ndarray) -> float:
            # L2 normalized effort
            effort = np.sum(weights * ((u - u0) / scales) ** 2)
            pred_sec = self._predict(u, current_state)
            # Soft penalty for missing target
            sec_penalty = 1500.0 * max(0.0, pred_sec - target_sec) ** 2
            return float(effort + sec_penalty)

        res = minimize(
            loss,
            u0,
            method="SLSQP",
            bounds=opt_bounds,
            options={"maxiter": 100, "ftol": 1e-6},
        )

        u_best = res.x if res.success else u0
        achieved_sec = self._predict(u_best, current_state)
        actual_reduction = max(0.0, curr_sec - achieved_sec)

        # Economic calculation (INR per day)
        # 1 Gcal = 3.9683 MMBTU, $11.50/MMBTU, 83.5 INR/USD
        daily_savings_inr = (
            actual_reduction * daily_urea_production_mt * 3.9683 * 11.50 * 83.5
        )

        # Construct actions
        deltas = u_best - u0
        actions: List[RecourseAction] = []
        feature_keys = ["steam_carbon_ratio", "reformer_outlet_temp", "loop_pressure"]

        # Sort features by absolute normalized effort
        norm_efforts = [abs(deltas[i] / scales[i]) for i in range(3)]
        ranked_indices = np.argsort(norm_efforts)[::-1]

        rank = 1
        summary_clauses = []
        for idx in ranked_indices:
            feat = feature_keys[idx]
            d = deltas[idx]
            if abs(d) > 1e-3:
                actions.append(
                    RecourseAction(
                        feature=feat,
                        current_value=round(float(u0[idx]), 3 if idx == 0 else 2),
                        recommended_target=round(float(u_best[idx]), 3 if idx == 0 else 2),
                        adjustment_delta=round(float(d), 3 if idx == 0 else 2),
                        units=self.FEATURE_UNITS[feat],
                        effort_rank=rank,
                    )
                )
                direction = "increase" if d > 0 else "decrease"
                summary_clauses.append(f"{direction} {feat} by {abs(d):.2f} {self.FEATURE_UNITS[feat]}")
                rank += 1

        total_effort = float(np.sum(norm_efforts))

        if summary_clauses:
            summary = (
                f"To save ₹{daily_savings_inr:,.0f}/day ({actual_reduction:.3f} Gcal/MT SEC reduction), "
                + ", and ".join(summary_clauses)
                + ". All adjustments remain strictly within 1-hour slew rate governors."
            )
        else:
            summary = "Plant is already operating at local thermodynamic optimum within governor bounds."

        return CounterfactualResult(
            current_sec_gcal_per_mt=round(curr_sec, 4),
            target_sec_gcal_per_mt=round(target_sec, 4),
            achievable_sec_gcal_per_mt=round(achieved_sec, 4),
            energy_reduction_gcal_per_mt=round(actual_reduction, 4),
            estimated_daily_savings_inr=round(daily_savings_inr, 2),
            total_recourse_effort_score=round(total_effort, 3),
            action_plan=actions,
            operator_executive_summary=summary,
        )
