"""Market report: live data -> forecast distribution -> quantum vs classical -> backtest.

Usage:
    python scripts/market_report.py --ticker AAPL
    python scripts/market_report.py --ticker NVDA --days 60 --qubits 5
    python scripts/market_report.py --ticker AAPL --demo     # simulated market, no network

Writes figures to results/figures/ and a summary to results/market_<TICKER>.txt.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from qstock.backtest import coverage_band, run_backtest
from qstock.forecast import Forecast, point_grid_prob_above, quantum_call_price, quantum_prob_above
from qstock.market import (
    atm_implied_vol,
    bs_call_price,
    load_calls,
    load_snapshot,
    pick_expiry,
    year_fraction,
)
from qstock.charts import backtest_chart, forecast_chart, smile_chart
from qstock.plotting import apply_style

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "results" / "figures"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--days", type=int, default=30, help="minimum days to the option expiry")
    ap.add_argument("--qubits", type=int, default=5, help="qubits for the price grid (2**n points)")
    ap.add_argument("--epsilon", type=float, default=0.005, help="IAE target precision")
    ap.add_argument("--up", type=float, default=0.05, help="move to test, e.g. 0.05 = +5%%")
    ap.add_argument("--demo", action="store_true", help="use a simulated market (no network)")
    args = ap.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)
    apply_style()
    out: list[str] = []

    def say(line: str = "") -> None:
        print(line)
        out.append(line)

    # ------------------------------------------------------------ 1. current market
    snap = load_snapshot(args.ticker, demo=args.demo)
    expiry = pick_expiry(snap, args.days)
    T = year_fraction(expiry, snap.asof)
    calls = load_calls(snap, expiry)
    iv = atm_implied_vol(calls, snap.spot)
    hv = snap.hist_vol

    tag = "DEMO DATA (simulated)" if snap.source == "demo" else "Yahoo Finance, ~15 min delayed"
    say(f"{snap.ticker}  as of {snap.asof:%Y-%m-%d}  [{tag}]")
    say(f"  spot            {snap.spot:10.2f}")
    say(f"  risk-free rate  {snap.rate:10.2%}")
    say(f"  expiry          {expiry}  ({T * 365:.0f} days)")
    say(f"  historical vol  {hv:10.1%}   (last 252 trading days)")
    say(f"  implied vol     {iv:10.1%}   (at-the-money, from option mid prices)")
    say("  -> market expects " + ("MORE" if iv > hv else "LESS") + " movement than the past year")

    # ------------------------------------------------------------ 2. forecast
    fcs = {"historical": Forecast(snap.spot, hv, snap.rate, T), "implied": Forecast(snap.spot, iv, snap.rate, T)}
    target = snap.spot * (1 + args.up)
    say()
    say(f"Forecast for {expiry}")
    for name, fc in fcs.items():
        lo, hi = fc.interval(0.9)
        say(f"  [{name:10s}] 90% range {lo:8.2f} - {hi:8.2f}   P(> {target:.2f}) = {fc.prob_above(target):6.1%}")

    q = quantum_prob_above(fcs["implied"], target, args.qubits, args.epsilon, seed=7)
    say(
        f"  [quantum IAE ] P(> {target:.2f}) = {q.value:6.1%}  "
        f"(95% CI {q.ci_low:.1%}-{q.ci_high:.1%}, circuit exact {q.exact:.1%}, {q.a_calls:,} calls to A)"
    )
    say(f"  [point grid  ] P(> {target:.2f}) = {point_grid_prob_above(fcs['implied'], target, args.qubits):6.1%}"
        f"  (naive {2**args.qubits}-point sampling, for comparison)")

    # ------------------------------------------------------------ 3. prices
    near = calls[(calls["strike"] > snap.spot * 0.9) & (calls["strike"] < snap.spot * 1.1)].dropna(subset=["mid"])
    near = near.iloc[:: max(1, len(near) // 5)].head(5)
    say()
    say(f"Call prices, {args.qubits}-qubit grid, IAE epsilon={args.epsilon}")
    say(f"  {'strike':>8} {'market':>8} {'BS(IV)':>8} {'BS(HV)':>8} {'quantum':>8} {'95% CI':>17}")
    for k, mid in zip(near["strike"], near["mid"]):
        qc = quantum_call_price(fcs["implied"], float(k), args.qubits, args.epsilon, seed=11)
        say(
            f"  {k:8.2f} {mid:8.2f} {bs_call_price(snap.spot, k, snap.rate, iv, T):8.2f} "
            f"{bs_call_price(snap.spot, k, snap.rate, hv, T):8.2f} {qc.value:8.2f} "
            f"   [{qc.ci_low:6.2f},{qc.ci_high:6.2f}]"
        )
    say("  market = bid/ask mid; BS(IV) differs from market away from the money because of the volatility smile")

    # ------------------------------------------------------------ 4. backtest
    horizon = max(5, round(T * 252))
    bt = run_backtest(snap.history, horizon=horizon)
    say()
    say(f"Backtest: {horizon}-trading-day forecasts from trailing 1y vol, {len(bt.records)} dates "
        f"(~{bt.independent_n} independent)")
    for _, row in bt.coverage.iterrows():
        lo, hi = coverage_band(row.level, bt.independent_n)
        flag = "ok" if lo <= row.observed <= hi else ("overconfident" if row.observed < lo else "too wide")
        say(f"  {row.level:4.0%} interval contained the outcome {row.observed:6.1%}   "
            f"(calibrated range {lo:.0%}-{hi:.0%})  {flag}")
    tail = float((bt.records["z"].abs() > 3).mean())
    say(f"  moves beyond 3 sigma: {tail:.2%} (normal model: 0.27%)")

    # ------------------------------------------------------------ figures
    for name, fig in [
        ("forecast", forecast_chart(snap, fcs, expiry, target)),
        ("backtest", backtest_chart(snap, bt)),
        ("smile", smile_chart(snap, calls, hv, expiry)),
    ]:
        fig.savefig(FIGURES / f"{name}_{snap.ticker}.png")
        plt.close(fig)
    (ROOT / "results" / f"market_{snap.ticker}.txt").write_text("\n".join(out) + "\n")
    print(f"\nFigures saved to {FIGURES}")


if __name__ == "__main__":
    main()
