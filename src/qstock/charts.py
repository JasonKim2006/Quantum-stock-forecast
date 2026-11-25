"""Figures shared by scripts/market_report.py and the Streamlit dashboard (app.py).

Each function returns a matplotlib Figure; callers save or display it.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .plotting import INK_2, SERIES, apply_style

apply_style()


def forecast_chart(snap, fcs, expiry, target):
    """Last 6 months of prices plus a sqrt(t)-shaped forecast cone to expiry.

    Implied-vol bands are filled (50% darker, 90% lighter); the historical-vol
    90% band is drawn as dashed lines so the two never blend into one colour.
    """
    from .forecast import Forecast

    fig, ax = plt.subplots(figsize=(8, 4.2))
    hist = snap.history.iloc[-126:]
    ax.plot(hist.index, hist.values, color=INK_2, lw=1.5, label="last 6 months")

    t0, t1 = hist.index[-1], np.datetime64(expiry)
    frac = np.linspace(0, 1, 40)
    times = t0 + (t1 - np.datetime64(t0)) * frac
    T = fcs["implied"].T

    def band(fc, level):
        lo, hi = [], []
        for f in frac:
            if f == 0:
                lo.append(snap.spot); hi.append(snap.spot); continue
            sub = Forecast(fc.spot, fc.vol, fc.rate, T * f)
            a, b = sub.interval(level)
            lo.append(a); hi.append(b)
        return np.array(lo), np.array(hi)

    imp = fcs["implied"]
    lo9, hi9 = band(imp, 0.9)
    lo5, hi5 = band(imp, 0.5)
    ax.fill_between(times, lo9, hi9, color=SERIES[0], alpha=0.15, lw=0, label="implied vol: 90% range")
    ax.fill_between(times, lo5, hi5, color=SERIES[0], alpha=0.35, lw=0, label="implied vol: 50% range")
    hlo, hhi = band(fcs["historical"], 0.9)
    ax.plot(times, hlo, color=SERIES[1], ls="--", lw=1.4, label="historical vol: 90% range")
    ax.plot(times, hhi, color=SERIES[1], ls="--", lw=1.4)
    ax.axhline(target, color=SERIES[3], ls=":", lw=1.4, label=f"target {target:.2f}")
    ax.set_title(f"{snap.ticker}: where the price may be on {expiry}")
    ax.set_ylabel("price")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    return fig


def backtest_chart(snap, bt):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    a1.plot([0, 1], [0, 1], color=INK_2, ls="--", lw=1)
    a1.plot(bt.coverage["expected"], bt.coverage["observed"], marker="o", color=SERIES[0])
    a1.set(xlabel="forecast interval level", ylabel="share of outcomes inside", title="Calibration",
           xlim=(0.4, 1.0), ylim=(0.4, 1.0))
    a2.hist(bt.records["pit"], bins=10, range=(0, 1), color=SERIES[0], alpha=0.8, density=True)
    a2.axhline(1, color=INK_2, ls="--", lw=1)
    a2.set(xlabel="PIT (forecast CDF at outcome)", ylabel="density", title="Should be flat")
    fig.suptitle(f"{snap.ticker}: {bt.horizon}-day forecast backtest", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


def smile_chart(snap, calls, hv, expiry):
    ok = calls.dropna(subset=["iv"])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(ok["strike"], ok["iv"], marker="o", color=SERIES[0], label="implied vol")
    ax.axhline(hv, color=SERIES[1], ls="--", label="historical vol")
    ax.axvline(snap.spot, color=INK_2, lw=1, label="spot")
    ax.set(xlabel="strike", ylabel="volatility", title=f"{snap.ticker} volatility smile, {expiry}")
    ax.legend()
    return fig


def convergence_chart(q_calls, q_err, mc_calls, mc_err):
    """RMS error vs calls to the sampler, log-log, with fitted slopes in the legend."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, (x, y, name, mk) in enumerate(
        [(mc_calls, mc_err, "Monte Carlo", "o"), (q_calls, q_err, "Quantum IAE", "s")]
    ):
        x, y = np.asarray(x, float), np.asarray(y, float)
        ok = y > 0
        slope = np.polyfit(np.log10(x[ok]), np.log10(y[ok]), 1)[0] if ok.sum() > 1 else np.nan
        ax.loglog(x, y, marker=mk, color=SERIES[i], label=f"{name}  (slope {slope:.2f})")
    ax.set(xlabel="calls to the sampler", ylabel="RMS error",
           title="Error vs cost (theory: Monte Carlo -0.5, quantum -1)")
    ax.legend()
    return fig
