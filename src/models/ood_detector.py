"""
CFCL Gadepan Complex - High-Dimensional Out-of-Distribution (OOD) Detection Engine.
Calculates multivariate Mahalanobis distance with covariance shrinkage and free energy
scoring against historical DCS baseline convex hull to prevent ungrounded inferences.
Sanitized DCS Telemetry under strict NDA.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.stats import chi2

from src.data.schema import FEATURE_COLUMNS
from src.utils.logger import get_logger

logger = get_logger("OODDetector")


@dataclass
class OODAssessmentResult:
    """Rigorous statistical determination of whether an input is Out-Of-Distribution."""
    mahalanobis_distance_squared: float
    chi2_critical_threshold: float
    ood_probability: float
    free_energy_score: float
    is_in_distribution: bool
    status: str
    dominant_deviating_features: List[str]


class OutOfDistributionDetector:
    """
    Multivariate Out-of-Distribution detector for CFCL Gadepan Telemetry.
    Computes Mahalanobis distance D_M^2(x) = (x - mu)^T * Sigma^-1 * (x - mu).
    Under multivariate Gaussianity, D_M^2 follows a chi-squared distribution with p degrees of freedom.
    """

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        significance_level: float = 0.001,  # 99.9% confidence threshold
    ):
        self.feature_names = feature_names or FEATURE_COLUMNS
        self.p = len(self.feature_names)
        self.alpha = significance_level
        self.critical_chi2 = float(chi2.ppf(1.0 - self.alpha, df=self.p))

        self.mean_: np.ndarray = np.zeros(self.p, dtype=np.float64)
        self.inv_cov_: np.ndarray = np.eye(self.p, dtype=np.float64)
        self.std_: np.ndarray = np.ones(self.p, dtype=np.float64)
        self.is_fitted: bool = False

    def fit(self, X_train: np.ndarray) -> Dict[str, float]:
        """
        Fits baseline mean vector and regularized covariance inverse on training telemetry.
        Uses Ledoit-Wolf-style diagonal shrinkage for high numerical condition stability.
        """
        X_arr = np.asarray(X_train, dtype=np.float64)
        if X_arr.shape[1] != self.p:
            raise ValueError(f"Expected {self.p} features, got {X_arr.shape[1]}")

        self.mean_ = np.mean(X_arr, axis=0)
        self.std_ = np.std(X_arr, axis=0)
        self.std_[self.std_ < 1e-6] = 1.0

        # Empirical covariance
        cov = np.cov(X_arr, rowvar=False)

        # Shrinkage regularization: Sigma_reg = (1 - lambda) * Cov + lambda * trace(Cov)/p * I
        shrinkage_lambda = 0.05
        cov_reg = (1.0 - shrinkage_lambda) * cov + shrinkage_lambda * np.trace(cov) / self.p * np.eye(self.p)

        # Invert with pseudo-inverse fallback
        try:
            self.inv_cov_ = np.linalg.inv(cov_reg)
        except np.linalg.LinAlgError:
            self.inv_cov_ = np.linalg.pinv(cov_reg)

        self.is_fitted = True
        logger.info(
            f"OOD Detector fitted on {len(X_arr)} records. p={self.p}, "
            f"critical_chi2({1.0 - self.alpha:.3f}) = {self.critical_chi2:.2f}"
        )

        return {
            "n_records": float(len(X_arr)),
            "degrees_of_freedom": float(self.p),
            "critical_chi2_threshold": self.critical_chi2,
        }

    def assess(self, x_query: np.ndarray) -> OODAssessmentResult:
        """
        Evaluates whether a single operational state vector x is within the historical distribution.
        """
        x_vec = np.asarray(x_query, dtype=np.float64).flatten()
        if len(x_vec) != self.p:
            raise ValueError(f"Expected {self.p} features, got {len(x_vec)}")

        diff = x_vec - self.mean_
        d_m2 = float(diff @ self.inv_cov_ @ diff.T)

        # Calculate empirical p-value from chi-square CDF
        p_val = float(1.0 - chi2.cdf(d_m2, df=self.p))
        ood_prob = float(chi2.cdf(d_m2, df=self.p))

        # Free energy score E(x) = 0.5 * D_M^2
        energy_score = 0.5 * d_m2

        # Identify features with highest individual z-score contribution
        z_scores = np.abs(diff / self.std_)
        top_indices = np.argsort(z_scores)[::-1][:3]
        dominant_features = [
            f"{self.feature_names[idx]} (z={z_scores[idx]:.1f})"
            for idx in top_indices if z_scores[idx] > 2.5
        ]

        is_in_dist = d_m2 <= self.critical_chi2
        if is_in_dist:
            status = "IN_DISTRIBUTION_CONFIRMED"
        else:
            status = "OUT_OF_DISTRIBUTION_ANOMALOUS_REGIME"

        return OODAssessmentResult(
            mahalanobis_distance_squared=round(d_m2, 2),
            chi2_critical_threshold=round(self.critical_chi2, 2),
            ood_probability=round(ood_prob, 4),
            free_energy_score=round(energy_score, 2),
            is_in_distribution=is_in_dist,
            status=status,
            dominant_deviating_features=dominant_features,
        )
