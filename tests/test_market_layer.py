"""Checks for the market, forecast, quantum-encoding and backtest code (no network)."""

import numpy as np
import pandas as pd
import pytest

from qstock.backtest import run_backtest
from qstock.forecast import Forecast, cell_model, quantum_call_price, quantum_prob_above
from qstock.market import bs_call_price, implied_vol, load_calls, load_snapshot, pick_expiry
from qstock.quantum import build_exact_problem, good_state_probability


def test_implied_vol_round_trip():
    price = bs_call_price(100, 105, 0.04, 0.27, 0.25)
    assert implied_vol(float(price), 100, 105, 0.04, 0.25) == pytest.approx(0.27, abs=1e-6)


def test_implied_vol_rejects_arbitrage_prices():
    assert np.isnan(implied_vol(0.0, 100, 105, 0.04, 0.25))
    assert np.isnan(implied_vol(150.0, 100, 105, 0.04, 0.25))


def test_exact_encoding_matches_expectation():
    rng = np.random.default_rng(0)
    pmf = rng.random(16)
    pmf /= pmf.sum()
    f = 3 * rng.random(16)
    pr = build_exact_problem(pmf, f)
    assert pr.post_processing(good_state_probability(pr)) == pytest.approx(pmf @ f, rel=1e-10)


def test_cell_model_reproduces_closed_form():
    fc = Forecast(spot=100, vol=0.3, rate=0.04, T=0.1)
    _, pmf, cond, covered = cell_model(fc, lambda s: (s > 104).astype(float), n_qubits=4, kinks=(104,))
    assert covered * pmf @ cond == pytest.approx(fc.prob_above(104), abs=1e-4)


def test_quantum_estimates_agree_with_formulas():
    fc = Forecast(spot=100, vol=0.3, rate=0.04, T=0.1)
    p = quantum_prob_above(fc, 104, n_qubits=4, epsilon=0.002, seed=1)
    assert p.ci_low - 2e-3 <= fc.prob_above(104) <= p.ci_high + 2e-3
    c = quantum_call_price(fc, 100, n_qubits=4, epsilon=0.002, seed=1)
    assert c.value == pytest.approx(float(bs_call_price(100, 100, 0.04, 0.3, 0.1)), rel=0.03)


def test_demo_market_end_to_end():
    snap = load_snapshot("AAPL", demo=True)
    calls = load_calls(snap, pick_expiry(snap, 20))
    assert snap.source == "demo"
    assert calls["iv"].notna().sum() > 5


def test_backtest_is_calibrated_on_true_gbm():
    rng = np.random.default_rng(3)
    n = 252 * 12
    closes = pd.Series(
        100 * np.exp(np.cumsum(0.25 / np.sqrt(252) * rng.standard_normal(n))),
        index=pd.bdate_range("2010-01-01", periods=n),
    )
    bt = run_backtest(closes, horizon=21, step=21)
    obs90 = bt.coverage.set_index("level").loc[0.9, "observed"]
    assert 0.83 <= obs90 <= 0.96


def test_every_string_has_both_languages():
    from qstock.i18n import STR

    assert all(set(v) == {"ko", "en"} for v in STR.values())


def test_web_charts_build():
    from qstock.web_charts import fan_chart

    snap = load_snapshot("AAPL", demo=True)
    fc = Forecast(snap.spot, 0.3, snap.rate, 30 / 365)
    end = snap.asof + pd.Timedelta(days=30)
    for lang in ("ko", "en"):
        fig = fan_chart(lang, snap, fc, fc, end, snap.spot * 1.05, 0.3)
        assert len(fig.data) >= 6


def test_implied_distribution_is_consistent():
    from qstock.implied import market_forecast
    from qstock.market import load_puts

    snap = load_snapshot("SPY", demo=True)
    e = pick_expiry(snap, 25)
    T = (pd.Timestamp(e) - snap.asof.normalize()).days / 365
    mf = market_forecast(load_calls(snap, e), load_puts(snap, e), snap.spot, snap.rate, T)
    # forward from parity, and the density's mean equals the forward (martingale)
    assert mf.dist.smile.forward == pytest.approx(snap.spot * np.exp(snap.rate * T), rel=1e-4)
    assert mf.dist.mean() == pytest.approx(mf.dist.smile.forward, rel=2e-3)
    assert mf.dist.clipped_mass < 0.01
    # the quantum estimate works on the implied distribution too
    q = quantum_prob_above(mf, snap.spot * 1.03, n_qubits=4, epsilon=0.002, seed=3)
    assert q.ci_low - 2e-3 <= mf.prob_above(snap.spot * 1.03) <= q.ci_high + 2e-3


def test_stock_forecast_shape_and_quantum():
    from qstock.stockdist import stock_forecast

    snap = load_snapshot("AAPL", demo=True)
    f = stock_forecast(snap.history, 30 / 365, snap.rate)
    lo, hi = f.interval(0.9)
    assert lo < snap.spot < hi
    assert f.interval_at(0.9, 0.0) == pytest.approx((snap.spot, snap.spot))
    assert f.interval_at(0.9, 1.0)[1] == pytest.approx(hi, rel=0.01)
    q = quantum_prob_above(f, snap.spot * 1.05, n_qubits=4, epsilon=0.002, seed=2)
    assert q.ci_low - 3e-3 <= f.prob_above(snap.spot * 1.05) <= q.ci_high + 3e-3


def test_fhs_backtest_calibrated_on_fat_tailed_garch():
    """On data with volatility clustering and fat tails, FHS should beat the bell curve in the tails."""
    rng = np.random.default_rng(5)
    n, var, r = 252 * 10, 0.0001, []
    for _ in range(n):  # GARCH(1,1) with Student-t shocks
        z = rng.standard_t(4) / np.sqrt(2)
        r.append(np.sqrt(var) * z)
        var = 2e-6 + 0.08 * r[-1] ** 2 + 0.90 * var
    closes = pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range("2012-01-02", periods=n))
    fhs = run_backtest(closes, horizon=5, step=5, model="fhs")
    bell = run_backtest(closes, horizon=5, step=5, model="normal")
    tail = lambda b: float((b.records["z"].abs() > 3).mean())  # noqa: E731
    assert tail(fhs) < tail(bell)
    assert abs(float(fhs.coverage.set_index("level").loc[0.9, "observed"]) - 0.9) < 0.04
