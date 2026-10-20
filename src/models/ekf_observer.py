"""
CFCL Gadepan Complex - Extended Kalman Filter (EKF) State Observer.
Tracks unobservable physical degradation states:
  1. Catalyst effectiveness factor (eta_catalyst)
  2. Cooling water heat exchanger fouling resistance (R_f)
  3. Primary reformer tube coking pressure drop (Delta_P_coking)
Sanitized DCS Telemetry under strict NDA.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np

from src.utils.logger import get_logger

logger = get_logger("EKFObserver")


@dataclass
class EKFStateEstimate:
    """Estimated internal unmeasured states from EKF filter."""
    catalyst_effectiveness_factor: float
    cw_fouling_resistance_m2k_per_w: float
    reformer_coking_dp_bar: float
    state_uncertainties: Dict[str, float]
    innovation_residual_norm: float
    is_healthy_regime: bool
    diagnostic_commentary: str


class ExtendedKalmanFilterObserver:
    """
    Non-linear state observer fusing DCS sensor telemetry with physical process models.

    States:
      x[0]: eta_catalyst  (Nominal = 1.0, aged = 0.75 - 0.85)
      x[1]: R_f_cw        (Nominal clean = 0.0002, fouled = 0.0008 m^2*K/W)
      x[2]: dp_coking     (Nominal clean = 1.6, coked = 3.2 bar)

    Measurements:
      z[0]: Observed SEC (Gcal/MT)
      z[1]: CO2 slip ppm (or CH4 slip %)
      z[2]: Reformer process differential pressure (bar)
    """

    def __init__(self):
        # Initial state estimates: [eta, R_f, dp_coking]
        self.x = np.array([0.96, 0.00030, 1.85], dtype=np.float64)

        # State covariance matrix P
        self.P = np.diag([0.02**2, (0.0001)**2, 0.15**2])

        # Process noise covariance Q (slow drift / aging per hour)
        self.Q = np.diag([1e-7, 1e-9, 1e-6])

        # Measurement noise covariance R (sensor noise)
        self.R = np.diag([0.02**2, 12.0**2, 0.05**2])

    def _observation_model(
        self,
        x: np.ndarray,
        operating_inputs: Dict[str, float],
    ) -> np.ndarray:
        """
        h(x, u): Non-linear physical relationship between unobserved states and sensor measurements.
        """
        eta = x[0]
        r_f = x[1]
        dp_coking = x[2]

        sc = float(operating_inputs.get("steam_carbon_ratio", 3.05))
        temp = float(operating_inputs.get("reformer_outlet_temp", 798.0))
        cw_in = float(operating_inputs.get("cw_inlet_temp", 31.5))

        # Expected SEC: lower eta increases SEC; higher R_f increases SEC
        # Base ~ 5.40 at 800 C, 3.05 S/C, eta=1.0, R_f=0.0002
        pred_sec = (
            5.40
            - 0.0035 * (temp - 800.0)
            - 0.12 * (sc - 3.05)
            + 0.015 * (cw_in - 30.0)
            + 0.45 * (1.0 - eta)
            + 180.0 * (r_f - 0.0002)
        )

        # Expected CO2/CH4 slip: lower eta increases slip
        pred_slip = 450.0 + 850.0 * (1.0 - eta) - 5.5 * (temp - 800.0)

        # Expected reformer differential pressure: base clean dP ~ 1.5 bar + coking dP
        pred_dp = dp_coking + 0.005 * (temp - 800.0)

        return np.array([pred_sec, pred_slip, pred_dp], dtype=np.float64)

    def _compute_jacobian_h(
        self,
        x: np.ndarray,
        operating_inputs: Dict[str, float],
    ) -> np.ndarray:
        """Numerical / analytical Jacobian of h(x, u) with respect to state x."""
        h0 = self._observation_model(x, operating_inputs)
        H = np.zeros((3, 3), dtype=np.float64)
        eps = 1e-5

        for j in range(3):
            x_pert = x.copy()
            x_pert[j] += eps
            h_pert = self._observation_model(x_pert, operating_inputs)
            H[:, j] = (h_pert - h0) / eps

        return H

    def step(
        self,
        measurements: Dict[str, float],
        operating_inputs: Dict[str, float],
    ) -> EKFStateEstimate:
        """
        Executes one recursive Predict-Update cycle of the Extended Kalman Filter.
        """
        # 1. Predict Step (Random walk state propagation)
        x_pred = self.x.copy()
        P_pred = self.P + self.Q

        # 2. Measurement Vector z
        z_obs = np.array(
            [
                float(measurements.get("sec_gcal_per_mt", 5.45)),
                float(measurements.get("co2_slip_ppm", 480.0)),
                float(measurements.get("reformer_dp_bar", 1.95)),
            ],
            dtype=np.float64,
        )

        # 3. Measurement Model & Innovation
        z_pred = self._observation_model(x_pred, operating_inputs)
        innovation = z_obs - z_pred
        innov_norm = float(np.linalg.norm(innovation))

        # 4. Jacobian Matrix H
        H = self._compute_jacobian_h(x_pred, operating_inputs)

        # 5. Kalman Gain K = P * H^T * (H * P * H^T + R)^-1
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.inv(S)

        # 6. State & Covariance Update
        self.x = x_pred + K @ innovation
        # Physical bounds clamping
        self.x[0] = max(0.65, min(1.10, self.x[0]))       # eta
        self.x[1] = max(0.0001, min(0.0020, self.x[1]))   # R_f
        self.x[2] = max(1.0, min(5.0, self.x[2]))         # dP coking

        I_mat = np.eye(3)
        self.P = (I_mat - K @ H) @ P_pred

        std_devs = np.sqrt(np.diag(self.P))

        # Diagnostic classification
        is_healthy = (self.x[0] >= 0.82) and (self.x[1] <= 0.0007) and (self.x[2] <= 2.8)
        diag_msgs = []
        if self.x[0] < 0.82:
            diag_msgs.append(f"Catalyst deactivation detected (eta={self.x[0]:.2f})")
        if self.x[1] > 0.0007:
            diag_msgs.append(f"Cooling water condenser fouling (R_f={self.x[1]:.5f})")
        if self.x[2] > 2.8:
            diag_msgs.append(f"Reformer tube carbon coking (dP={self.x[2]:.2f} bar)")
        if not diag_msgs:
            diag_msgs.append("Internal states operating in pristine healthy design envelope.")

        return EKFStateEstimate(
            catalyst_effectiveness_factor=round(float(self.x[0]), 3),
            cw_fouling_resistance_m2k_per_w=float(f"{self.x[1]:.5f}"),
            reformer_coking_dp_bar=round(float(self.x[2]), 2),
            state_uncertainties={
                "eta_std": round(float(std_devs[0]), 4),
                "rf_std": float(f"{std_devs[1]:.6f}"),
                "dp_std": round(float(std_devs[2]), 4),
            },
            innovation_residual_norm=round(innov_norm, 3),
            is_healthy_regime=is_healthy,
            diagnostic_commentary="; ".join(diag_msgs),
        )
