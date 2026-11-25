"""The market's own forecast: the risk-neutral distribution implied by option prices.

Breeden & Litzenberger (1978): if C(K) is the price of a call with strike K and
expiry T, the risk-neutral density of S_T is

    q(K) = e^{rT} * d^2 C / dK^2.

Differentiating noisy quotes twice directly is unstable, so the usual practical
recipe is used:

1. **Forward.** Put-call parity, C - P = e^{-rT}(F - K), gives the forward price
   F at each strike where both quotes exist; the median of the strikes nearest
   the money is used. This absorbs dividends and borrow costs automatically.
2. **Smile.** Out-of-the-money quotes (puts below F, calls above), which are the
   liquid ones, are converted to Black implied vols and a low-order polynomial
   in log-moneyness k = log(K/F) is fitted. Outside the quoted strikes the total
   variance vol^2 T grows linearly in k, as Lee's moment formula allows (see Smile.vol).
3. **Density.** Call prices on a fine strike grid come from Black's formula with
   the fitted smile; the second derivative gives q. Small negative values (a sign
   of slight arbitrage in the fit) are clipped and the density renormalised; the
   clipped mass is reported.

The result is a distribution with the skew and fat left tail that real equity
options price in, which a single-volatility lognormal cannot show.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm


def black_call(F, K, r, T, vol):
    F, K, vol = np.asarray(F, float), np.asarray(K, float), np.asarray(vol, float)
    s = vol * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * s**2) / s
    return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d1 - s))


def black_iv(price: float, F: float, K: float, r: float, T: float, call: bool) -> float:
    """Black implied vol of a call (or a put, converted by parity); NaN if not solvable."""
    if not np.isfinite(price) or price <= 0:
        return float("nan")
    c = price if call else price + np.exp(-r * T) * (F - K)
    intrinsic = np.exp(-r * T) * max(F - K, 0.0)
    if c <= intrinsic + 1e-8 or c >= np.exp(-r * T) * F:
        return float("nan")
    try:
        return float(brentq(lambda v: black_call(F, K, r, T, v) - c, 1e-4, 5.0))
    except ValueError:
        return float("nan")


def estimate_forward(calls: pd.DataFrame, puts: pd.DataFrame, spot: float, r: float, T: float) -> float:
    both = calls[["strike", "mid"]].merge(puts[["strike", "mid"]], on="strike", suffixes=("_c", "_p")).dropna()
    if both.empty:
        return float(spot * np.exp(r * T))
    both = both.iloc[(both["strike"] - spot).abs().argsort()[:4]]
    fwd = both["strike"] + np.exp(r * T) * (both["mid_c"] - both["mid_p"])
    return float(np.median(fwd))


@dataclass
class Smile:
    forward: float
    T: float
    rate: float
    k: np.ndarray        # log-moneyness of the quotes used
    vols: np.ndarray     # their Black implied vols
    coeffs: np.ndarray   # polynomial in k
    k_lo: float
    k_hi: float

    def vol(self, k):
        """Fitted vol inside the quoted range, and a smooth wing outside it.

        Beyond the last quote, total variance w = vol^2 * T starts with exactly the
        fitted value, slope and curvature (no kink, so no spike or step in the
        density). Its slope then eases toward a far-wing value clipped to [0, 1.9],
        since Roger Lee's moment formula only allows w to grow linearly in |k| with
        slope at most 2. Bending the smile flat right at the last strike, or forcing
        its slope to zero there, does create such artefacts.
        """
        k = np.asarray(k, float)
        T = self.T
        d = np.polyder(self.coeffs)
        v = np.polyval(self.coeffs, np.clip(k, self.k_lo, self.k_hi))
        out = v.copy()
        d2 = np.polyder(d)
        L = max(0.05, 0.25 * (self.k_hi - self.k_lo))
        for edge, sign in ((self.k_lo, -1.0), (self.k_hi, 1.0)):
            v_e, dv, d2v = (np.polyval(p, edge) for p in (self.coeffs, d, d2))
            # Total variance w = v^2 T and its first two derivatives along the outward direction.
            w_e = v_e**2 * T
            s0 = float(sign * 2 * v_e * dv * T)                  # outward slope at the edge
            c0 = float(2 * (dv**2 + v_e * d2v) * T)               # curvature at the edge
            s_inf = float(np.clip(s0, 0.0, 1.9))                  # far-wing slope Lee's bound allows
            gap = np.maximum(sign * (k - edge), 0.0)
            fade = np.exp(-gap / L)
            # w(0)=w_e, w'(0)=s0, w''(0)=c0 exactly (no kink, no step in the density); far out the
            # slope settles at s_inf, so the wing grows linearly within Lee's bound.
            c = c0 + (s0 - s_inf) / L
            w = w_e + s_inf * gap + (s0 - s_inf) * L * (1 - fade) + 0.5 * c * gap**2 * fade
            w = np.maximum(w, 1e-4 * T)
            out = np.where(gap > 0, np.sqrt(w / T), out)
        return np.clip(out, 0.01, 3.0)

    @property
    def atm_vol(self) -> float:
        return float(self.vol(0.0))


def fit_smile(calls: pd.DataFrame, puts: pd.DataFrame, spot: float, r: float, T: float) -> Smile:
    F = estimate_forward(calls, puts, spot, r, T)
    otm = pd.concat(
        [
            puts.loc[puts["strike"] < F, ["strike", "mid", "bid", "ask"]].assign(call=False),
            calls.loc[calls["strike"] >= F, ["strike", "mid", "bid", "ask"]].assign(call=True),
        ]
    ).dropna(subset=["mid"])
    # Drop quotes whose spread is wider than the price itself: too noisy to use.
    otm = otm[(otm["ask"] - otm["bid"]) <= otm["mid"]]
    iv = np.array([black_iv(m, F, K, r, T, c) for m, K, c in zip(otm["mid"], otm["strike"], otm["call"])])
    k = np.log(otm["strike"].to_numpy(float) / F)
    ok = np.isfinite(iv) & (np.abs(k) < 1.5)
    k, iv = k[ok], iv[ok]
    if len(k) < 4:
        raise ValueError("Not enough option quotes to fit a smile; try during market hours.")
    deg = 3 if len(k) >= 10 else 2
    coeffs = np.polyfit(k, iv, deg)
    return Smile(F, T, r, k, iv, coeffs, float(k.min()), float(k.max()))


@dataclass
class MarketDistribution:
    """Discretised risk-neutral distribution of S_T on a fine grid."""

    grid: np.ndarray
    pdf_vals: np.ndarray
    cdf_vals: np.ndarray
    clipped_mass: float
    smile: Smile = field(repr=False)

    @classmethod
    def from_smile(cls, smile: Smile, n: int = 4001) -> "MarketDistribution":
        width = float(np.clip(8 * smile.atm_vol * np.sqrt(smile.T), 0.2, 3.0))
        k = np.linspace(-width, width, n)
        K = smile.forward * np.exp(k)
        C = black_call(smile.forward, K, smile.rate, smile.T, smile.vol(k))
        q = np.exp(smile.rate * smile.T) * np.gradient(np.gradient(C, K), K)
        neg = float(-np.trapezoid(np.minimum(q, 0), K))
        q = np.maximum(q, 0)
        total = float(np.trapezoid(q, K))
        q = q / total
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (q[1:] + q[:-1]) * np.diff(K))])
        cdf = cdf / cdf[-1]
        return cls(K, q, cdf, neg / (total + neg), smile)

    def cdf(self, x):
        return np.interp(x, self.grid, self.cdf_vals, left=0.0, right=1.0)

    def sf(self, x):
        return 1.0 - self.cdf(x)

    def pdf(self, x):
        return np.interp(x, self.grid, self.pdf_vals, left=0.0, right=0.0)

    def ppf(self, q):
        # cdf can be flat in the far tails; np.unique keeps interp well defined.
        c, idx = np.unique(self.cdf_vals, return_index=True)
        return np.interp(q, c, self.grid[idx])

    def median(self) -> float:
        return float(self.ppf(0.5))

    def mean(self) -> float:
        return float(np.trapezoid(self.grid * self.pdf_vals, self.grid))


@dataclass(frozen=True)
class MarketForecast:
    """Same interface as :class:`qstock.forecast.Forecast`, backed by the implied distribution."""

    spot: float
    rate: float
    T: float
    dist: MarketDistribution

    @property
    def vol(self) -> float:  # used only for labels and swing estimates
        return self.dist.smile.atm_vol

    def interval(self, level: float = 0.9) -> tuple[float, float]:
        a = (1 - level) / 2
        return float(self.dist.ppf(a)), float(self.dist.ppf(1 - a))

    def prob_above(self, x: float) -> float:
        return float(self.dist.sf(x))

    def support(self, tail: float = 3e-5) -> tuple[float, float]:
        return float(self.dist.ppf(tail)), float(self.dist.ppf(1 - tail))

    # The market only prices the expiry; intermediate dates scale log-distances by sqrt(t),
    # which is exact for a lognormal and a reasonable picture otherwise.
    def interval_at(self, level: float, frac: float) -> tuple[float, float]:
        lo, hi = self.interval(level)
        g = np.sqrt(max(frac, 0.0))
        return float(self.spot * (lo / self.spot) ** g), float(self.spot * (hi / self.spot) ** g)

    def median_at(self, frac: float) -> float:
        return float(self.spot * (self.dist.median() / self.spot) ** np.sqrt(max(frac, 0.0)))


MAX_CLIPPED = 0.03
MAX_MEAN_GAP = 0.01


def market_forecast(calls: pd.DataFrame, puts: pd.DataFrame, spot: float, r: float, T: float) -> MarketForecast:
    """Fit the smile and build the implied distribution.

    Raises ValueError when the quotes are too sparse, or the fit implies more than
    MAX_CLIPPED negative probability or a mean more than MAX_MEAN_GAP away from the
    forward; callers then fall back to the single-volatility lognormal.
    """
    smile = fit_smile(calls, puts, spot, r, T)
    dist = MarketDistribution.from_smile(smile)
    if dist.clipped_mass > MAX_CLIPPED:
        raise ValueError(f"Implied density unreliable ({dist.clipped_mass:.1%} negative mass).")
    # Under the pricing measure E[S_T] must equal the forward; a large gap means a bad fit.
    if abs(dist.mean() / smile.forward - 1) > MAX_MEAN_GAP:
        raise ValueError(f"Implied density unreliable (mean {dist.mean():.2f} vs forward {smile.forward:.2f}).")
    return MarketForecast(spot, r, T, dist)
