"""
CFCL Gadepan Complex - Monte Carlo Gas Tariff & FX Volatility Value-at-Risk Engine.
Runs 10,000-trial stochastic simulations across Indian pooled gas tariffs (APM + RLNG blend),
exchange rate fluctuations, and turnaround risk to establish CFO-grade VaR99 / CVaR99 bounds.
Sanitized DCS Telemetry under strict NDA.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from src.utils.logger import get_logger

logger = get_logger("MonteCarloRisk")


@dataclass
class MonteCarloRiskReport:
    """Stochastic financial risk metrics for executive investment committees."""
    trials: int
    mean_annual_savings_inr_cr: float
    median_annual_savings_inr_cr: float
    std_annual_savings_inr_cr: float
    var_95_inr_cr: float
    var_99_inr_cr: float
    cvar_99_inr_cr: float
    percentile_5th_inr_cr: float
    percentile_95th_inr_cr: float
    gas_tariff_simulated_mean_usd: float
    exchange_rate_simulated_mean_inr: float
    downside_tail_risk_probability: float
    financial_resilience_rating: str


class MonteCarloSavingsSimulator:
    """
    Simulates stochastic market conditions for CFCL Gadepan:
      1. Domestic APM Gas Cap ($6.50/MMBTU, capped by Kirit Parikh committee guidelines)
      2. Imported RLNG spot & term contracts (LogNormal distribution)
      3. Domestic gas allocation blend ratio w_domestic ~ Beta(12, 8) (~60% domestic)
      4. USD/INR exchange rate volatility ~ Normal(83.50, 2.10)
    """

    MMBTU_PER_GCAL = 3.96832

    def __init__(self, base_annual_urea_production_mt: float = 1750.0 * 335.0):
        self.annual_production_mt = base_annual_urea_production_mt

    def simulate(
        self,
        mean_delta_sec_gcal_mt: float = 0.038,
        n_trials: int = 10000,
        random_seed: int = 42,
    ) -> MonteCarloRiskReport:
        """
        Executes 10,000 Monte Carlo paths of gas price and foreign exchange.
        """
        np.random.seed(random_seed)

        # 1. Domestic APM Gas Price ($/MMBTU) ~ Truncated Normal around $6.50 cap
        apm_price = np.random.normal(loc=6.50, scale=0.35, size=n_trials)
        apm_price = np.clip(apm_price, 5.0, 7.50)

        # 2. Imported RLNG Price ($/MMBTU) ~ LogNormal (reflecting global LNG spikes)
        # Median $13.20, sigma=0.18
        mu_rlng = np.log(13.20)
        rlng_price = np.random.lognormal(mean=mu_rlng, sigma=0.18, size=n_trials)

        # 3. Blend weighting: w_domestic ~ Beta(12, 8) -> Mean ~ 0.60
        w_domestic = np.random.beta(a=12.0, b=8.0, size=n_trials)

        # 4. Pooled Gas Supply Price ($/MMBTU)
        pooled_gas_usd = w_domestic * apm_price + (1.0 - w_domestic) * rlng_price

        # 5. Foreign Exchange Rate (INR / USD) ~ Normal(83.50, 2.10)
        fx_inr_usd = np.random.normal(loc=83.50, scale=2.10, size=n_trials)
        fx_inr_usd = np.clip(fx_inr_usd, 78.0, 92.0)

        # 6. Operational Delta SEC variability ~ Normal(mean, 0.004)
        sec_realized = np.random.normal(loc=mean_delta_sec_gcal_mt, scale=0.0035, size=n_trials)
        sec_realized = np.clip(sec_realized, 0.015, 0.065)

        # Annual energy saved (Gcal)
        energy_saved_gcal = sec_realized * self.annual_production_mt
        energy_saved_mmbtu = energy_saved_gcal * self.MMBTU_PER_GCAL

        # Annual monetary savings in INR Crores
        # INR / trial = energy_saved_mmbtu * pooled_gas_usd * fx_inr_usd / 1e7
        savings_crore = (energy_saved_mmbtu * pooled_gas_usd * fx_inr_usd) / 1e7

        # Value-at-Risk & Conditional VaR calculations
        mean_val = float(np.mean(savings_crore))
        median_val = float(np.median(savings_crore))
        std_val = float(np.std(savings_crore))

        # Downside percentiles (5th and 1st percentile of savings)
        p5 = float(np.percentile(savings_crore, 5.0))
        p95 = float(np.percentile(savings_crore, 95.0))
        p1 = float(np.percentile(savings_crore, 1.0))

        # VaR_99 is the drop from the mean to the 1st percentile: VaR_alpha = Mean - Percentile_(1-alpha)
        var_95 = mean_val - p5
        var_99 = mean_val - p1

        # CVaR_99 (Expected Shortfall): Mean of all trials falling below the 1st percentile
        cvar_99 = mean_val - float(np.mean(savings_crore[savings_crore <= p1]))

        # Downside tail risk: Probability of annual savings falling below ₹12.0 Crore
        prob_under_12cr = float(np.mean(savings_crore < 12.0))

        rating = "AAA_SOVEREIGN_GRADE_RESILIENCE" if p1 >= 10.0 else "INVESTMENT_GRADE"

        logger.info(
            f"Monte Carlo ({n_trials} trials): Mean=₹{mean_val:.2f} Cr, "
            f"VaR99=₹{var_99:.2f} Cr, CVaR99=₹{cvar_99:.2f} Cr"
        )

        return MonteCarloRiskReport(
            trials=n_trials,
            mean_annual_savings_inr_cr=round(mean_val, 2),
            median_annual_savings_inr_cr=round(median_val, 2),
            std_annual_savings_inr_cr=round(std_val, 2),
            var_95_inr_cr=round(var_95, 2),
            var_99_inr_cr=round(var_99, 2),
            cvar_99_inr_cr=round(cvar_99, 2),
            percentile_5th_inr_cr=round(p5, 2),
            percentile_95th_inr_cr=round(p95, 2),
            gas_tariff_simulated_mean_usd=round(float(np.mean(pooled_gas_usd)), 2),
            exchange_rate_simulated_mean_inr=round(float(np.mean(fx_inr_usd)), 2),
            downside_tail_risk_probability=round(prob_under_12cr, 4),
            financial_resilience_rating=rating,
        )
