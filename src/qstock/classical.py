"""Classical reference prices: Black-Scholes, Monte Carlo, and the discretised model."""

from __future__ import annotations

import numpy as np
from scipy.stats import lognorm, norm

from .params import OptionParams


def black_scholes_call(p: OptionParams) -> float:
    """Closed-form price of a European call."""
    d1 = (np.log(p.S0 / p.K) + (p.r + 0.5 * p.vol**2) * p.T) / (p.vol * np.sqrt(p.T))
    d2 = d1 - p.vol * np.sqrt(p.T)
    return float(p.S0 * norm.cdf(d1) - p.K * p.discount * norm.cdf(d2))


def mc_call_price(p: OptionParams, n_samples: int, rng: np.random.Generator) -> float:
    """Plain Monte Carlo price using exact sampling of S_T (no time stepping)."""
    z = rng.standard_normal(n_samples)
    s_T = np.exp(p.mu + p.sigma * z)
    return float(p.discount * np.maximum(s_T - p.K, 0.0).mean())


def grid_and_pmf(p: OptionParams, n_qubits: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniform grid of 2**n_qubits points on the truncation interval and its pmf.

    This is the distribution the quantum circuit loads into the amplitudes of an
    n-qubit register: the log-normal pdf sampled on the grid and renormalised.
    """
    low, high = p.bounds
    x = np.linspace(low, high, 2**n_qubits)
    pdf = lognorm.pdf(x, s=p.sigma, scale=np.exp(p.mu))
    return x, pdf / pdf.sum()


def discretised_call_price(p: OptionParams, n_qubits: int) -> float:
    """Exact expectation of the payoff under the discretised, truncated model.

    This is the value an ideal (noise-free, infinitely precise) quantum
    algorithm would converge to, ignoring payoff-encoding approximations.
    """
    x, pmf = grid_and_pmf(p, n_qubits)
    return float(p.discount * np.dot(pmf, np.maximum(x - p.K, 0.0)))
