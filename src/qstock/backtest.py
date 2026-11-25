"""Does the forecast distribution actually hold up? A walk-forward check.

At every test date t we use only data up to t:

1. build the forecast for log(S_{t+h} / S_t): a bell curve with the trailing
   volatility (model="normal"), or the filtered historical simulation of
   qstock.stockdist (model="fhs");
2. wait h days and record where the real outcome fell (its PIT, the forecast's
   CDF at the outcome).

If the forecast is *calibrated*, a 90% interval contains the outcome about 90%
of the time, and the PIT values (forecast CDF at the outcome) are uniform on
[0, 1]. Too few hits means the model was overconfident; too many means it was
too cautious.

Zero drift is used because estimating a stock's expected return from one
year of data is far noisier than the drift itself. Both models use price
history only, so both can be backtested; option-implied forecasts could not be,
because Yahoo does not provide historical option data.

Consecutive test dates overlap when ``step < horizon``, so the effective
number of independent forecasts is roughly ``len(results) * step / horizon``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

LEVELS = (0.5, 0.8, 0.9, 0.95)


@dataclass
class BacktestResult:
    horizon: int
    lookback: int
    step: int
    records: pd.DataFrame   # date, vol_ann, z, pit, realised_return
    coverage: pd.DataFrame  # level, expected, observed, n, independent_n

    @property
    def independent_n(self) -> int:
        return int(len(self.records) * min(self.step, self.horizon) / self.horizon)


def run_backtest(
    closes: pd.Series, horizon: int = 21, lookback: int = 252, step: int = 5, model: str = "normal",
    n_sims: int = 4000, seed: int = 0,
) -> BacktestResult:
    """Walk-forward check of one model.

    model="normal": bell curve with the trailing ``lookback``-day volatility.
    model="fhs": filtered historical simulation (see qstock.stockdist), rebuilt at
    each date from the returns and EWMA volatility known on that date only.
    """
    from .stockdist import ewma_vol, simulate_log_returns

    logp = np.log(closes.to_numpy(float))
    r = np.diff(logp)
    dates = closes.index
    rng = np.random.default_rng(seed)
    if model == "fhs":
        sigma = ewma_vol(r)  # sigma[t] uses returns before t only, so it is known on date t
        z_all = r / sigma[:-1]
    rows = []
    for t in range(lookback, len(logp) - horizon, step):
        realised = logp[t + horizon] - logp[t]
        if model == "fhs":
            pool = z_all[20:t] - z_all[20:t].mean()
            sims = simulate_log_returns(pool, sigma[t], r[:t].std(ddof=1), horizon, n_sims, rng)[:, -1]
            pit = (np.sum(sims < realised) + 0.5 * np.sum(sims == realised) + 0.5) / (n_sims + 1)
            vol_ann = sigma[t] * np.sqrt(252)
        else:
            sig_d = np.diff(logp[t - lookback : t + 1]).std(ddof=1)
            pit = norm.cdf(realised / (sig_d * np.sqrt(horizon)))
            vol_ann = sig_d * np.sqrt(252)
        rows.append(
            {
                "date": dates[t],
                "vol_ann": vol_ann,
                "z": norm.ppf(np.clip(pit, 1e-6, 1 - 1e-6)),  # outcome in 'standard deviations' of the forecast
                "pit": pit,
                "realised_return": np.expm1(realised),
            }
        )
    rec = pd.DataFrame(rows)
    if rec.empty:
        raise ValueError("Not enough history for this lookback and horizon.")

    n_ind = int(len(rec) * min(step, horizon) / horizon)
    cov = pd.DataFrame(
        [
            {
                "level": lv,
                "expected": lv,
                "observed": float(((rec["pit"] - 0.5).abs() <= lv / 2).mean()),
                "n": len(rec),
                "independent_n": n_ind,
            }
            for lv in LEVELS
        ]
    )
    return BacktestResult(horizon, lookback, step, rec, cov)


def coverage_band(level: float, n_independent: int, z: float = 1.96) -> tuple[float, float]:
    """Rough 95% range of observed coverage for a *calibrated* model."""
    se = np.sqrt(level * (1 - level) / max(n_independent, 1))
    return max(0.0, level - z * se), min(1.0, level + z * se)
