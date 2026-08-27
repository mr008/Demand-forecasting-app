"""Lognormal demand distribution fitted to forecast quantiles.

The forecasting models emit quantiles; the risk and order stages need a
continuous distribution to compute stock-out probabilities, service levels and
expected lost sales. A lognormal fitted to (q50, q90) is a pragmatic choice for
non-negative, right-skewed demand.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

Z90 = 1.2815515655446004
MIN_SIGMA = 0.05
MAX_SIGMA = 1.5  # q90/q50 ratio capped at ~6.8x: keeps near-zero-demand lines from exploding
MIN_MEDIAN = 1e-6
SIGMA_FLOOR_UNITS = 1.0  # sigma is estimated with the median floored at one unit


def lognormal_params(q50: np.ndarray, q90: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(mu, sigma) of ln(D) such that the median is q50 and the 90th percentile q90."""
    q50 = np.maximum(np.asarray(q50, dtype=float), MIN_MEDIAN)
    q90 = np.maximum(np.asarray(q90, dtype=float), q50)
    base = np.maximum(q50, SIGMA_FLOOR_UNITS)
    sigma = np.clip(np.log(np.maximum(q90, base) / base) / Z90, MIN_SIGMA, MAX_SIGMA)
    return np.log(q50), sigma


def prob_demand_exceeds(stock: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """P(D > stock); 1.0 when stock <= 0."""
    stock = np.asarray(stock, dtype=float)
    with np.errstate(divide="ignore"):
        z = (np.log(np.maximum(stock, MIN_MEDIAN)) - mu) / sigma
    p = 1.0 - norm.cdf(z)
    return np.where(stock <= 0, 1.0, p)


def quantile(mu: np.ndarray, sigma: np.ndarray, level: np.ndarray | float) -> np.ndarray:
    return np.exp(mu + sigma * norm.ppf(level))


def mean(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return np.exp(mu + sigma**2 / 2)


def expected_shortfall(stock: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """E[max(D - stock, 0)] for lognormal D (units of lost sales over the period)."""
    stock = np.asarray(stock, dtype=float)
    s = np.maximum(stock, MIN_MEDIAN)
    with np.errstate(divide="ignore"):
        ln_s = np.log(s)
    m = mean(mu, sigma)
    term1 = m * (1.0 - norm.cdf((ln_s - mu - sigma**2) / sigma))
    term2 = s * (1.0 - norm.cdf((ln_s - mu) / sigma))
    out = term1 - term2
    return np.where(stock <= 0, m, np.maximum(out, 0.0))
