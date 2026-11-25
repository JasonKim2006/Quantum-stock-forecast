"""Forecast distribution built from the stock's own price history (no options needed).

Filtered Historical Simulation (FHS; Barone-Adesi et al., 1999)
--------------------------------------------------------------
1. **Today's volatility.** An EWMA of squared daily log returns (RiskMetrics,
   lambda = 0.94) tracks how jumpy the stock is *right now*, rather than averaging
   the past year equally.
2. **Shape from history.** Each past return is divided by the EWMA volatility
   just before it, giving standardised shocks z. Their distribution keeps the real
   fat tails and asymmetry; only the size is removed.
3. **Simulate.** Future daily returns are shocks drawn with replacement from z
   and rescaled by a volatility path that starts at today's level and decays toward
   the long-run average (half-life 30 trading days). Summing h days gives the log
   return to the forecast date; 20,000 paths give the distribution.

No bell-curve assumption is made anywhere, so crashes and spikes appear about
as often as they did in the stock's own history, and a calm or a stormy market
widens or narrows the forecast automatically. Drift is set to zero: estimating
a stock's expected return from a few years of data is far noisier than the
drift itself.

The result has the same interface as :class:`qstock.forecast.Forecast`, so the
fan chart, the quantum circuit and the IBM runs work with it unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

LAMBDA = 0.94
HALF_LIFE = 30
N_PATHS = 20_000
TRADING_DAYS = 252


def ewma_vol(log_ret: np.ndarray, lam: float = LAMBDA, warmup: int = 20) -> np.ndarray:
    """sigma[t] = EWMA daily vol known *before* return t (sigma[len] is tomorrow's)."""
    var = np.empty(len(log_ret) + 1)
    var[0] = np.var(log_ret[:warmup]) if len(log_ret) >= warmup else np.var(log_ret)
    for i, x in enumerate(log_ret):
        var[i + 1] = lam * var[i] + (1 - lam) * x * x
    return np.sqrt(var)


def shocks(log_ret: np.ndarray, sigma: np.ndarray, warmup: int = 20) -> np.ndarray:
    """Standardised, demeaned shocks z = r / sigma_before."""
    z = log_ret[warmup:] / sigma[warmup:-1]
    return z - z.mean()


def vol_path(sigma_now: float, sigma_long: float, h: int) -> np.ndarray:
    """Daily vol for each of the next h days, decaying from now toward the long-run level."""
    decay = 0.5 ** (np.arange(h) / HALF_LIFE)
    return np.sqrt(sigma_long**2 + (sigma_now**2 - sigma_long**2) * decay)


def simulate_log_returns(z, sigma_now, sigma_long, h, n, rng) -> np.ndarray:
    """(n, h) cumulative log returns."""
    draws = rng.choice(z, size=(n, h)) * vol_path(sigma_now, sigma_long, h)
    return np.cumsum(draws, axis=1)


@dataclass
class EmpiricalDistribution:
    """Smoothed distribution of S_T from simulated samples, on a fine price grid."""

    grid: np.ndarray
    pdf_vals: np.ndarray
    cdf_vals: np.ndarray
    samples: np.ndarray = field(repr=False)

    @classmethod
    def from_log_returns(cls, spot: float, x: np.ndarray, n_grid: int = 1500) -> "EmpiricalDistribution":
        # Gaussian kernel smoothing in log-return space (Silverman bandwidth): a smooth CDF
        # the quantum cell model can integrate, without changing the sample's shape.
        bw = 1.06 * x.std() * len(x) ** (-0.2)
        lo, hi = x.min() - 4 * bw, x.max() + 4 * bw
        y = np.linspace(lo, hi, n_grid)
        sub = x
        cdf = np.zeros(n_grid)
        pdf = np.zeros(n_grid)
        for chunk in np.array_split(sub, max(1, len(sub) // 1000)):
            u = (y[:, None] - chunk[None, :]) / bw
            cdf += norm.cdf(u).sum(axis=1)
            pdf += norm.pdf(u).sum(axis=1)
        cdf /= len(sub)
        pdf /= len(sub) * bw
        price = spot * np.exp(y)
        return cls(price, pdf / price, cdf, spot * np.exp(x))  # density in price space

    def cdf(self, s):
        return np.interp(s, self.grid, self.cdf_vals, left=0.0, right=1.0)

    def sf(self, s):
        return 1.0 - self.cdf(s)

    def pdf(self, s):
        return np.interp(s, self.grid, self.pdf_vals, left=0.0, right=0.0)

    def ppf(self, q):
        c, idx = np.unique(self.cdf_vals, return_index=True)
        return np.interp(q, c, self.grid[idx])

    def median(self) -> float:
        return float(self.ppf(0.5))

    def mean(self) -> float:
        return float(self.samples.mean())


@dataclass(frozen=True)
class StockForecast:
    spot: float
    rate: float
    T: float
    dist: EmpiricalDistribution
    vol: float                      # today's EWMA vol, annualised
    long_vol: float                 # long-run vol, annualised
    n_shocks: int                   # how many past daily moves the shapes come from
    path_q: np.ndarray = field(repr=False)   # (n_levels, h+1) price quantiles per day
    levels: np.ndarray = field(repr=False)

    def interval(self, level: float = 0.9) -> tuple[float, float]:
        a = (1 - level) / 2
        return float(self.dist.ppf(a)), float(self.dist.ppf(1 - a))

    def prob_above(self, x: float) -> float:
        return float(self.dist.sf(x))

    def support(self, tail: float = 3e-5) -> tuple[float, float]:
        return float(self.dist.ppf(tail)), float(self.dist.ppf(1 - tail))

    def _q_at(self, q: float, frac: float) -> float:
        """Quantile q of the price a fraction ``frac`` of the way to T, from the simulated paths."""
        day = frac * (self.path_q.shape[1] - 1)
        d0 = int(np.floor(day))
        d1 = min(d0 + 1, self.path_q.shape[1] - 1)
        v0, v1 = (np.interp(q, self.levels, self.path_q[:, d]) for d in (d0, d1))
        return float(v0 + (v1 - v0) * (day - d0))

    def interval_at(self, level: float, frac: float) -> tuple[float, float]:
        a = (1 - level) / 2
        return self._q_at(a, frac), self._q_at(1 - a, frac)

    def median_at(self, frac: float) -> float:
        return self._q_at(0.5, frac)


def stock_forecast(closes: pd.Series, T: float, rate: float = 0.0, n_paths: int = N_PATHS,
                   seed: int = 0) -> StockForecast:
    """FHS forecast of the price ``T`` years ahead from daily closes alone."""
    logp = np.log(closes.to_numpy(float))
    r = np.diff(logp)
    if len(r) < 120:
        raise ValueError("Need at least about six months of daily prices.")
    sigma = ewma_vol(r)
    z = shocks(r, sigma)
    sigma_now, sigma_long = float(sigma[-1]), float(r.std(ddof=1))
    h = max(1, int(round(T * TRADING_DAYS)))
    rng = np.random.default_rng(seed)
    cum = simulate_log_returns(z, sigma_now, sigma_long, h, n_paths, rng)
    spot = float(closes.iloc[-1])
    levels = np.linspace(0.005, 0.995, 199)
    q = np.quantile(cum, levels, axis=0)                       # (levels, h)
    path_q = np.hstack([np.full((len(levels), 1), spot), spot * np.exp(q)])  # column 0 is today
    return StockForecast(
        spot=spot,
        rate=rate,
        T=T,
        dist=EmpiricalDistribution.from_log_returns(spot, cum[:, -1]),
        vol=sigma_now * np.sqrt(TRADING_DAYS),
        long_vol=sigma_long * np.sqrt(TRADING_DAYS),
        n_shocks=len(z),
        path_q=path_q,
        levels=levels,
    )
