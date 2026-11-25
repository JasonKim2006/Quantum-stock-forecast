"""Live market data: spot, price history, risk-free rate and option chains.

Real data comes from Yahoo Finance through ``yfinance`` (free, unofficial,
delayed ~15 min). Because it can be unavailable (weekends, rate limits, no
network), every loader also has a ``demo`` mode that produces data of exactly
the same shape from a simulated market, clearly labelled as such.

Implied volatilities are recomputed here from bid/ask mid prices instead of
trusting Yahoo's ``impliedVolatility`` column, which is often stale or zero
outside market hours.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

TRADING_DAYS = 252
DEFAULT_RATE = 0.04


@dataclass
class MarketSnapshot:
    """Everything the forecast and pricing code needs about one underlying."""

    ticker: str
    spot: float
    history: pd.Series          # daily closes, DatetimeIndex
    rate: float                 # continuously compounded risk-free rate
    expiries: list[str]         # available option expiries, "YYYY-MM-DD"
    source: str                 # "yahoo" or "demo"
    asof: pd.Timestamp
    currency: str = "USD"

    @property
    def hist_vol(self) -> float:
        """Annualised volatility of daily log returns over the last year."""
        return realised_vol(self.history.iloc[-TRADING_DAYS - 1 :])


def realised_vol(closes: pd.Series) -> float:
    log_ret = np.log(closes / closes.shift(1)).dropna()
    return float(log_ret.std(ddof=1) * np.sqrt(TRADING_DAYS))


def year_fraction(expiry: str, asof: pd.Timestamp) -> float:
    """Calendar-day year fraction to expiry (options expire at the close)."""
    days = (pd.Timestamp(expiry) - asof.normalize()).days
    return max(days, 1) / 365.0


# ---------------------------------------------------------------- pricing helpers
def bs_call_price(S, K, r, vol, T):
    """Vectorised Black-Scholes call price."""
    S, K, vol = np.asarray(S, float), np.asarray(K, float), np.asarray(vol, float)
    d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_vol(price: float, S: float, K: float, r: float, T: float) -> float:
    """Volatility that reproduces ``price`` under Black-Scholes, or NaN."""
    intrinsic = max(S - K * np.exp(-r * T), 0.0)
    if not np.isfinite(price) or price <= intrinsic + 1e-6 or price >= S:
        return float("nan")
    try:
        return float(brentq(lambda v: bs_call_price(S, K, r, v, T) - price, 1e-4, 5.0))
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------- Yahoo loaders
def load_snapshot(ticker: str, demo: bool = False, years: int = 5) -> MarketSnapshot:
    """Current spot, ``years`` of daily closes, short rate and expiries."""
    if demo:
        return _demo_snapshot(ticker, years)

    import yfinance as yf

    tk = yf.Ticker(ticker)
    hist = tk.history(period=f"{years}y", auto_adjust=True)["Close"].dropna()
    if hist.empty:
        raise ValueError(f"No price history returned for {ticker!r}.")
    hist.index = hist.index.tz_localize(None)
    return MarketSnapshot(
        ticker=ticker.upper(),
        spot=float(hist.iloc[-1]),
        history=hist,
        rate=_load_rate(yf),
        expiries=list(tk.options),
        source="yahoo",
        asof=pd.Timestamp(hist.index[-1]),
        currency=_load_currency(tk),
    )


def _load_currency(tk) -> str:
    try:
        return str(tk.fast_info["currency"] or "USD").upper()
    except Exception:
        return "USD"


def _load_rate(yf) -> float:
    """13-week T-bill yield (^IRX, quoted in percent) as a continuous rate."""
    try:
        irx = yf.Ticker("^IRX").history(period="5d")["Close"].dropna()
        return float(np.log1p(irx.iloc[-1] / 100.0))
    except Exception:
        return DEFAULT_RATE


def load_calls(snap: MarketSnapshot, expiry: str) -> pd.DataFrame:
    """Call chain for one expiry with mid prices and recomputed implied vols.

    Columns: strike, bid, ask, mid, volume, open_interest, iv.
    Rows without a usable two-sided quote get ``iv = NaN``.
    """
    if snap.source == "demo":
        calls = _demo_calls(snap, expiry)
    else:
        import yfinance as yf

        raw = yf.Ticker(snap.ticker).option_chain(expiry).calls
        calls = pd.DataFrame(
            {
                "strike": raw["strike"].astype(float),
                "bid": raw["bid"].astype(float),
                "ask": raw["ask"].astype(float),
                "volume": raw["volume"].fillna(0).astype(float),
                "open_interest": raw["openInterest"].fillna(0).astype(float),
            }
        )

    calls["mid"] = np.where(
        (calls["bid"] > 0) & (calls["ask"] >= calls["bid"]),
        (calls["bid"] + calls["ask"]) / 2,
        np.nan,
    )
    T = year_fraction(expiry, snap.asof)
    calls["iv"] = [
        implied_vol(m, snap.spot, k, snap.rate, T) for m, k in zip(calls["mid"], calls["strike"])
    ]
    return calls.reset_index(drop=True)


def load_puts(snap: MarketSnapshot, expiry: str) -> pd.DataFrame:
    """Put chain for one expiry: strike, bid, ask, mid, volume, open_interest."""
    if snap.source == "demo":
        puts = _demo_quotes(snap, expiry, kind="put")
    else:
        import yfinance as yf

        raw = yf.Ticker(snap.ticker).option_chain(expiry).puts
        puts = pd.DataFrame(
            {
                "strike": raw["strike"].astype(float),
                "bid": raw["bid"].astype(float),
                "ask": raw["ask"].astype(float),
                "volume": raw["volume"].fillna(0).astype(float),
                "open_interest": raw["openInterest"].fillna(0).astype(float),
            }
        )
    puts["mid"] = np.where((puts["bid"] > 0) & (puts["ask"] >= puts["bid"]), (puts["bid"] + puts["ask"]) / 2, np.nan)
    return puts.reset_index(drop=True)


def atm_implied_vol(calls: pd.DataFrame, spot: float) -> float:
    """Implied vol interpolated to the spot (at-the-money) strike."""
    ok = calls.dropna(subset=["iv"])
    if ok.empty:
        raise ValueError("No strikes with a valid two-sided quote; try during market hours.")
    ok = ok.iloc[(ok["strike"] - spot).abs().argsort()[:4]].sort_values("strike")
    if len(ok) == 1:
        return float(ok["iv"].iloc[0])
    return float(np.interp(spot, ok["strike"], ok["iv"]))


def nearest_expiry(snap: MarketSnapshot, target: pd.Timestamp) -> str | None:
    """Listed expiry closest to ``target`` (None when the ticker has no options)."""
    if not snap.expiries:
        return None
    return min(snap.expiries, key=lambda e: abs((pd.Timestamp(e) - target).days))


def pick_expiry(snap: MarketSnapshot, min_days: int = 20) -> str:
    """First expiry at least ``min_days`` calendar days away."""
    for e in snap.expiries:
        if (pd.Timestamp(e) - snap.asof.normalize()).days >= min_days:
            return e
    return snap.expiries[-1]


# ---------------------------------------------------------------- demo market
_DEMO_SEED = {"AAPL": 1, "NVDA": 2, "SPY": 3, "MSFT": 4, "TSLA": 5}


def _demo_snapshot(ticker: str, years: int) -> MarketSnapshot:
    """Simulated GBM history ending 'today'; same shape as the Yahoo snapshot."""
    seed = _DEMO_SEED.get(ticker.upper(), sum(map(ord, ticker)))
    rng = np.random.default_rng(seed)
    n = years * TRADING_DAYS
    vol = 0.2 + 0.25 * rng.random()
    # Volatility regimes so the backtest has something to find.
    regime = 1.0 + 0.6 * (np.sin(np.linspace(0, 6 * np.pi, n)) > 0.7)
    rets = (0.08 - 0.5 * vol**2) / TRADING_DAYS + vol * regime * rng.standard_normal(n) / np.sqrt(
        TRADING_DAYS
    )
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    closes = pd.Series(100 * np.exp(np.cumsum(rets)), index=dates, name="Close")
    asof = dates[-1]
    expiries = [
        (asof + pd.Timedelta(days=d)).strftime("%Y-%m-%d") for d in (7, 14, 30, 60, 90, 180)
    ]
    return MarketSnapshot(
        ticker=ticker.upper(),
        spot=float(closes.iloc[-1]),
        history=closes,
        rate=DEFAULT_RATE,
        expiries=expiries,
        source="demo",
        asof=asof,
    )


def _demo_calls(snap: MarketSnapshot, expiry: str) -> pd.DataFrame:
    return _demo_quotes(snap, expiry, kind="call")


def _demo_quotes(snap: MarketSnapshot, expiry: str, kind: str) -> pd.DataFrame:
    """Quotes from Black-Scholes with a downside skew (like real equity options) and a spread."""
    T = year_fraction(expiry, snap.asof)
    base = 1.1 * snap.hist_vol
    # Listed strikes widen with maturity, roughly +-3 standard deviations of the move.
    span = float(np.clip(3 * base * np.sqrt(T), 0.25, 1.2))
    strikes = np.unique(np.round(snap.spot * np.exp(np.linspace(-span, span, 41)), 1))
    m = np.log(strikes / snap.spot)
    # Skew and curvature flatten with maturity (roughly like 1/sqrt(T)), as in real markets.
    # Without this a long-dated smile would violate Lee's tail bound and imply negative probability.
    vols = base + (0.8 * m**2 - 0.35 * m) * min(1.0, np.sqrt(30 / 365 / T))
    fair = bs_call_price(snap.spot, strikes, snap.rate, vols, T)
    if kind == "put":
        fair = fair - snap.spot + strikes * np.exp(-snap.rate * T)
    fair = np.maximum(fair, 0.0)
    half_spread = np.maximum(0.005, 0.02 * fair)
    return pd.DataFrame(
        {
            "strike": strikes,
            "bid": np.maximum(fair - half_spread, 0.0).round(3),
            "ask": (fair + half_spread).round(3),
            "volume": 100.0,
            "open_interest": 1000.0,
        }
    )
