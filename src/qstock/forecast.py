"""Probabilistic forecast of a future price, computed classically and on a quantum circuit.

What "forecast" means here
--------------------------
The model does **not** predict whether the price goes up or down. It predicts
a *distribution* for the price on a future date: a range and the probability
of landing in it. Under geometric Brownian motion,

    log S_T ~ Normal( log S_0 + (r - vol^2 / 2) T,  vol^2 T ).

Two choices of ``vol`` give two forecasts:

* **historical** -- how much the stock moved over the past year;
* **implied** -- the volatility that makes Black-Scholes match today's option
  prices, i.e. how much the market *expects* it to move.

The drift uses the risk-free rate (the pricing measure). Over weeks to a few
months the drift is small next to vol * sqrt(T), so the choice barely moves
the forecast; the backtest in :mod:`qstock.backtest` checks the result on data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import lognorm

from .classical import grid_and_pmf
from .params import OptionParams
from .quantum import ExpectationEstimate, build_exact_problem, estimate_expectation


@dataclass(frozen=True)
class Forecast:
    spot: float
    vol: float
    rate: float
    T: float

    @property
    def dist(self):
        mu = np.log(self.spot) + (self.rate - 0.5 * self.vol**2) * self.T
        return lognorm(s=self.vol * np.sqrt(self.T), scale=np.exp(mu))

    def interval(self, level: float = 0.9) -> tuple[float, float]:
        a = (1 - level) / 2
        return float(self.dist.ppf(a)), float(self.dist.ppf(1 - a))

    def prob_above(self, x: float) -> float:
        return float(self.dist.sf(x))

    def support(self, tail: float = 3e-5) -> tuple[float, float]:
        """Price range holding all but ``tail`` probability in each tail."""
        return float(self.dist.ppf(tail)), float(self.dist.ppf(1 - tail))

    def interval_at(self, level: float, frac: float) -> tuple[float, float]:
        """``interval(level)`` for the date a fraction ``frac`` of the way to T."""
        if frac <= 0:
            return self.spot, self.spot
        return Forecast(self.spot, self.vol, self.rate, self.T * frac).interval(level)

    def median_at(self, frac: float) -> float:
        if frac <= 0:
            return self.spot
        return float(Forecast(self.spot, self.vol, self.rate, self.T * frac).dist.median())

    def params(self, strike: float, n_std: float = 3.0) -> OptionParams:
        return OptionParams(S0=self.spot, K=strike, r=self.rate, vol=self.vol, T=self.T, n_std=n_std)


def cell_model(fc, payoff, n_qubits: int, kinks=(), sub: int = 256):
    """Discretise S_T into 2**n cells and the payoff into per-cell averages.

    Works for any forecast with ``support()`` and ``dist.cdf`` -- the lognormal
    :class:`Forecast` or the option-implied :class:`qstock.implied.MarketForecast`.

    The register state is sum_i sqrt(P(cell i)) |i>, and the rotation for cell
    i encodes E[f(S_T) | S_T in cell i] rather than f at one grid point. Then

        sum_i P(cell i) * E[f | cell i] = E[f(S_T); S_T in [low, high]]

    holds up to the sub-grid quadrature, so the model error left is the tail
    mass outside ``support()`` (6e-5 by default). With point sampling, a 32-point
    grid is off by percentage points for a digital payoff.

    Args:
        payoff: Vectorised f(S_T).
        kinks: Points where f jumps or bends (a strike, a threshold). They are
            added as sub-grid edges so the quadrature is exact there.

    Returns (cell midpoints, cell probabilities normalised to 1, per-cell
    payoff averages, probability mass covered by the grid).
    """
    d = fc.dist
    lo, hi = fc.support()
    edges = np.exp(np.linspace(np.log(lo), np.log(hi), 2**n_qubits + 1))
    fine = np.linspace(edges[:-1], edges[1:], sub + 1, axis=1).ravel()
    inside = [k for k in kinks if edges[0] < k < edges[-1]]
    fine = np.unique(np.concatenate([fine, inside]))
    w = np.diff(d.cdf(fine))
    mid = 0.5 * (fine[:-1] + fine[1:])
    f = payoff(mid)
    cell = np.clip(np.searchsorted(edges, mid) - 1, 0, len(edges) - 2)
    mass = np.bincount(cell, weights=w, minlength=len(edges) - 1)
    cond = np.bincount(cell, weights=w * f, minlength=len(edges) - 1) / np.where(mass > 0, mass, 1)
    covered = float(mass.sum())
    return 0.5 * (edges[:-1] + edges[1:]), mass / covered, cond, covered


def _estimate(fc, payoff, n_qubits, epsilon, seed, kinks=()) -> ExpectationEstimate:
    _, pmf, cond, covered = cell_model(fc, payoff, n_qubits, kinks)
    est = estimate_expectation(build_exact_problem(pmf, cond), epsilon=epsilon, seed=seed)
    c = covered
    return ExpectationEstimate(c * est.value, c * est.ci_low, c * est.ci_high, est.a_calls, c * est.exact)


def quantum_prob_above(
    fc, x: float, n_qubits: int = 5, epsilon: float = 0.005, seed: int | None = None
) -> ExpectationEstimate:
    """P(S_T > x) by amplitude estimation of the digital payoff 1[S_T > x]."""
    return _estimate(fc, lambda s: (s > x).astype(float), n_qubits, epsilon, seed, kinks=(x,))


def quantum_call_price(
    fc, strike: float, n_qubits: int = 5, epsilon: float = 0.005, seed: int | None = None
) -> ExpectationEstimate:
    """Discounted call price by amplitude estimation of max(S_T - K, 0)."""
    est = _estimate(fc, lambda s: np.maximum(s - strike, 0.0), n_qubits, epsilon, seed, kinks=(strike,))
    d = np.exp(-fc.rate * fc.T)
    return ExpectationEstimate(d * est.value, d * est.ci_low, d * est.ci_high, est.a_calls, d * est.exact)


def point_grid_prob_above(fc: Forecast, x: float, n_qubits: int = 5) -> float:
    """Same probability on the point-sampled grid used in the convergence study."""
    grid, pmf = grid_and_pmf(fc.params(strike=x), n_qubits)
    return float(pmf[grid > x].sum())
