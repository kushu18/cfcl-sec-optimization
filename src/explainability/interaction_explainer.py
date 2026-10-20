"""
CFCL Gadepan Complex - High-Order TreeSHAP Interaction Matrix Engine.
Computes full N x N SHAP pairwise interaction tensors showing non-linear thermodynamic
coupling across process variables (e.g., S/C Ratio x Reformer Outlet Temp).
Sanitized DCS Telemetry under strict NDA.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import shap

from src.data.schema import FEATURE_COLUMNS
from src.utils.logger import get_logger

logger = get_logger("InteractionExplainer")


@dataclass
class TopInteractionPair:
    """A pair of features with strong non-linear cross-coupling effect on SEC."""
    feature_a: str
    feature_b: str
    interaction_strength_gcal_mt: float
    description: str


@dataclass
class InteractionMatrixResult:
    """Complete N x N interaction matrix and top synergistic pairs."""
    feature_names: List[str]
    interaction_matrix: List[List[float]]
    top_interactions: List[TopInteractionPair]
    total_interaction_energy: float


class InteractionExplainer:
    """
    Computes exact TreeSHAP interaction values for tree ensembles (Random Forest, XGBoost).
    Decomposes model prediction into main effects and pairwise second-order interaction terms:
      f(x) - E[f(x)] = sum_i Phi_{i,i}(x) + 2 * sum_{i < j} Phi_{i,j}(x)
    """

    def __init__(self, model: Any, feature_names: Optional[List[str]] = None):
        self.model = model
        self.feature_names = feature_names or FEATURE_COLUMNS
        self.explainer: Optional[shap.TreeExplainer] = None

        try:
            # Check if model is supported by TreeExplainer
            self.explainer = shap.TreeExplainer(self.model)
        except Exception as e:
            logger.warning(f"Standard TreeExplainer initialization deferred: {e}")

    def compute_interaction_matrix(
        self,
        X_sample: np.ndarray,
        max_samples: int = 50,
    ) -> InteractionMatrixResult:
        """
        Computes the mean absolute interaction matrix over a representative batch of samples.
        """
        X_arr = np.asarray(X_sample, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        n_samples = min(len(X_arr), max_samples)
        X_eval = X_arr[:n_samples]
        n_features = len(self.feature_names)

        # If TreeExplainer available, compute exact shap_interaction_values
        if self.explainer is not None:
            try:
                # Shape: (n_samples, n_features, n_features)
                inter_vals = self.explainer.shap_interaction_values(X_eval)
                # Mean absolute interaction matrix across samples
                mean_matrix = np.mean(np.abs(inter_vals), axis=0)
            except Exception as e:
                logger.warning(f"TreeSHAP exact interaction computation fallback: {e}")
                mean_matrix = self._compute_numerical_interaction_matrix(X_eval)
        else:
            mean_matrix = self._compute_numerical_interaction_matrix(X_eval)

        # Zero out diagonal to focus purely on cross-variable coupling
        off_diag_matrix = mean_matrix.copy()
        np.fill_diagonal(off_diag_matrix, 0.0)

        # Extract top interaction pairs
        pairs: List[TopInteractionPair] = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                val = float(off_diag_matrix[i, j])
                feat_a = self.feature_names[i]
                feat_b = self.feature_names[j]

                # Process engineering contextualization
                desc = self._get_coupling_description(feat_a, feat_b)
                pairs.append(
                    TopInteractionPair(
                        feature_a=feat_a,
                        feature_b=feat_b,
                        interaction_strength_gcal_mt=round(val, 5),
                        description=desc,
                    )
                )

        pairs.sort(key=lambda p: p.interaction_strength_gcal_mt, reverse=True)
        total_energy = float(np.sum(off_diag_matrix))

        matrix_list = [
            [round(float(val), 5) for val in row] for row in mean_matrix
        ]

        return InteractionMatrixResult(
            feature_names=self.feature_names,
            interaction_matrix=matrix_list,
            top_interactions=pairs[:5],
            total_interaction_energy=round(total_energy, 4),
        )

    def _compute_numerical_interaction_matrix(self, X_eval: np.ndarray) -> np.ndarray:
        """Second-order partial difference numerical fallback for interaction matrix."""
        n_feat = len(self.feature_names)
        mat = np.zeros((n_feat, n_feat), dtype=np.float64)

        base_pred = np.mean(self.model.predict(X_eval))
        std_devs = np.std(X_eval, axis=0)
        std_devs[std_devs < 1e-6] = 1.0

        for i in range(n_feat):
            for j in range(i, n_feat):
                if i == j:
                    # Diagonal main effect
                    X_p = X_eval.copy()
                    X_p[:, i] += 0.5 * std_devs[i]
                    pred_p = np.mean(self.model.predict(X_p))
                    mat[i, i] = abs(pred_p - base_pred)
                else:
                    # Cross interaction: d^2 f / dx_i dx_j
                    X_pp = X_eval.copy()
                    X_pp[:, i] += 0.5 * std_devs[i]
                    X_pp[:, j] += 0.5 * std_devs[j]

                    X_pi = X_eval.copy()
                    X_pi[:, i] += 0.5 * std_devs[i]

                    X_pj = X_eval.copy()
                    X_pj[:, j] += 0.5 * std_devs[j]

                    cross_diff = (
                        self.model.predict(X_pp)
                        - self.model.predict(X_pi)
                        - self.model.predict(X_pj)
                        + self.model.predict(X_eval)
                    )
                    inter_strength = float(np.mean(np.abs(cross_diff)))
                    mat[i, j] = inter_strength
                    mat[j, i] = inter_strength

        return mat

    def _get_coupling_description(self, fa: str, fb: str) -> str:
        s = {fa, fb}
        if {"steam_carbon_ratio", "reformer_outlet_temp"} == s:
            return "Reformer Le Chatelier thermodynamic equilibrium coupling (CH4 conversion kinetics)."
        elif {"ambient_temp", "cw_inlet_temp"} == s:
            return "Cooling tower wet-bulb approach coupling (condenser vacuum & refrigeration power)."
        elif {"loop_pressure", "reformer_outlet_temp"} == s:
            return "Syngas compression vs synthesis loop single-pass ammonia conversion balance."
        elif {"steam_carbon_ratio", "loop_pressure"} == s:
            return "Hydrogen-to-inerts ratio in circulating loop gas vs reforming steam duty."
        else:
            return "Secondary non-linear thermodynamic cross-coupling."
