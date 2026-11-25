# Quantum Stock Forecast

A stock's own price history → a probabilistic forecast of its future price → the same
numbers computed on a quantum circuit with **Qiskit amplitude estimation** (in simulation
and on real IBM hardware), checked against closed-form answers, Monte Carlo, and a
walk-forward backtest.

The project began as quantum option pricing (amplitude estimation of a European call). That
work is still here: the convergence, bias and noise experiments in `scripts/`, and option-based
forecasts behind an advanced switch on the website.

> **What this does and doesn't predict.** No model here says whether a stock will go up or down.
> It forecasts a *distribution*: the range the price is likely to land in by a given
> date, and the probability of each part of that range. The backtest checks whether those
> ranges held up on real history. This is a learning project, not investment advice.

## What you get

| | |
|---|---|
| **Current market** | Spot price, 1-year historical volatility, the option chain with implied vols recomputed from bid/ask mids, the volatility smile, and the 13-week T-bill rate (Yahoo Finance via `yfinance`) |
| **Forecast** | 50% / 90% price ranges for 1 week to 1 year ahead, plus *P(price ends above X)*, built from **the stock's own price history** by filtered historical simulation (fat tails and today's volatility, no bell-curve assumption), with a bell curve for comparison. Option-based models are available behind an advanced switch |
| **Quantum** | The forecast distribution is loaded onto qubits, and Iterative Amplitude Estimation computes probabilities and call prices, with confidence intervals and a cost count |
| **IBM hardware** | The same circuit runs on a real IBM quantum computer (free Open Plan) or in a free local preview with an IBM device's noise model, with readout-error correction and job tracking |
| **Validation** | Quantum vs Black–Scholes vs market prices; walk-forward calibration backtest; error-vs-cost convergence (Monte Carlo slope ≈ −½, quantum ≈ −1) |
| **Website** | A bilingual (한국어 / English) Streamlit site: search a ticker, pick how far ahead, and read the answer as one sentence. Every line and shaded area in the charts explains itself on hover |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .

python run_dashboard.py                                # website at http://localhost:8501
python scripts/market_report.py --ticker AAPL          # text report + figures in results/figures/
python scripts/market_report.py --ticker NVDA --days 60 --up 0.10
python scripts/market_report.py --ticker AAPL --demo   # simulated market, works offline
pytest -q
```

Start the dashboard with `python run_dashboard.py`, not `streamlit run app.py`. Streamlit runs
the page in a new thread on every click. If Qiskit's compiled core is first imported in one of
those threads, the next one can segfault the server (seen with qiskit 2.6 / Python 3.13). The
launcher imports Qiskit in the main thread first.

Yahoo data is free, unofficial and about 15 minutes delayed. Outside market hours, bid/ask quotes are
often zero, which leaves no implied vols. Run during US market hours, or use demo mode.

## How it works

### 1. Forecast model
Under geometric Brownian motion,
`log S_T ~ Normal(log S_0 + (r − σ²/2)T, σ²T)`.
σ is either the **historical** vol (how much the stock moved over the past year) or the
**implied** vol (what option prices say the market expects). When implied is above historical,
the market is pricing in more movement than the stock has recently shown.

### 2. Quantum circuit (`src/qstock/quantum.py`, `src/qstock/forecast.py`)
1. **Load the distribution:** `StatePreparation` puts `Σᵢ √P(cell i) |i⟩` on *n* qubits (2ⁿ price cells, ±4σ in log-price).
2. **Encode the payoff:** a `UCRYGate` rotates one extra qubit by a cell-dependent angle, so
   `P(extra qubit = 1) = E[f(S_T)] / max f` *exactly*. The angle for each cell encodes the payoff
   **averaged over that cell**, not sampled at one grid point.
3. **Amplitude estimation:** `IterativeAmplitudeEstimation` (qiskit-algorithms) estimates that
   probability. Cost is counted in calls to the state-preparation circuit A: a shot at Grover power *k* costs 2k+1 calls.

The same circuit handles calls (`f = max(S−K, 0)`) and digital payoffs (`f = 1[S > X]`), which give
"probability the price ends above X".

**Cell averaging.** Sampling the payoff at 32 points, the usual textbook construction, biases a
"price above X" probability by about 1.6 percentage points in the AAPL demo (33.0% vs 31.4%). With cell
averaging, plus the strike or threshold inserted as an exact sub-grid edge, the circuit matches the
closed form to < 10⁻⁴, so any remaining gap is sampling error only.

### 3. Forecast from price history (`src/qstock/stockdist.py`), the default
Filtered Historical Simulation (Barone-Adesi et al., 1999):
1. **Today's volatility** from an EWMA of squared daily log returns (RiskMetrics, λ = 0.94).
2. **Shape from history**: each past return ÷ the EWMA volatility just before it gives a standardised shock.
   These keep the stock's real fat tails and asymmetry.
3. **Simulate** 20,000 paths: shocks drawn with replacement, scaled by a volatility path that starts at
   today's level and decays to the long-run level (half-life 30 trading days). Zero drift.
4. A Gaussian-kernel-smoothed CDF of the simulated prices feeds the same quantum circuit as before.

Because it needs only prices, it can be backtested exactly like the bell curve. On a fat-tailed
GARCH(1,1)/Student-t test series (`tests/test_market_layer.py`), 5-day forecasts:

| | 90% range hit rate | 95% range hit rate | moves beyond 3σ (ideal 0.27%) |
|---|---|---|---|
| Bell curve, 1-year vol | 89.0% | 91.6% | 1.55% |
| Filtered historical simulation | 89.6% | 94.0% | 0.22% |

### 4. The market's own distribution (`src/qstock/implied.py`, advanced)
Breeden & Litzenberger (1978): the risk-neutral density of S_T is `e^{rT} ∂²C/∂K²`. In practice:
1. **Forward** from put-call parity at the strikes nearest the money. This absorbs dividends.
2. **Smile**: out-of-the-money puts and calls → Black implied vols → cubic fit in log-moneyness.
3. **Wings**: beyond the last quote, total variance keeps the fitted value, slope *and* curvature, then
   eases to a linear slope in [0, 1.9] (Roger Lee's moment bound). Flat extrapolation creates a kink,
   which shows up as negative probability; a slope forced to zero creates a spike.
4. **Density** by differentiating Black call prices twice on a fine grid. Checks: E[S_T] must equal the
   forward (martingale) to within 1%, with < 3% negative mass clipped, or the site falls back to the bell curve.

On the demo market (which has a realistic downside skew), the market distribution gives a 20%+ fall a
3.5% chance, about twice the 1.8% from a bell curve with the same at-the-money vol.

### 5. IBM hardware (`src/qstock/ibm.py`)
One Sampler job holds the k=0 circuit (A, measure), optionally a k=1 circuit (A then one Grover
iteration), and two readout-calibration circuits on the same physical qubit
(`p = (p_meas − e0)/(1 − e0 − e1)`). A free **preview** runs the identical transpiled circuits locally
against `FakeFez`, a snapshot of ibm_fez's error rates. Results from that preview with 3 price qubits:

| circuit | 2-qubit gates | exact | measured (corrected) |
|---|---|---|---|
| k=0 | 14 | 33.5% | 33.3% |
| k=1 | 61 | 92.2% | 83.3% |

The direct estimate survives; one Grover step already loses about 9 points to noise (and about 20 at 4
qubits). This is why full amplitude estimation is not yet practical on hardware. Save an account once
with `QiskitRuntimeService.save_account(...)`. The key lives in `~/.qiskit`, never in this repo.

### 6. Backtest (`src/qstock/backtest.py`)
At each past date, each model forecasts the *h*-day move using only prices known on that date, then
the code checks where the price really landed. A calibrated model's 90% range contains the outcome
about 90% of the time, and its PIT histogram is flat. The report shows how many forecasts are
independent (overlapping windows are not), and the range of hit rates that still counts as calibrated.
Only historical vol can be backtested, because Yahoo has no historical option data.

## Results (demo market, 5 qubits, ε = 0.005)

```
[implied   ] P(> 29.63) = 31.4%
[quantum IAE] P(> 29.63) = 31.4%  (95% CI 31.3%–31.5%, 15,000 calls to A)
[point grid ] P(> 29.63) = 33.0%  (naive 32-point sampling)

strike  market  BS(IV)  quantum   95% CI
 27.00    1.95    1.93     1.93   [1.92, 1.95]
 28.00    1.35    1.34     1.33   [1.30, 1.36]

Backtest, 21-day horizon: 90% range contained the outcome 90.9% of the time (calibrated)
Convergence: Monte Carlo slope −0.52, quantum IAE slope −0.90
```

Run on a real ticker to replace these numbers with live ones. For real stocks, expect the backtest to
find **fat tails**: more 3σ moves than the normal model's 0.27%.

## Earlier experiments (same package)

| Script | Question |
|---|---|
| `scripts/run_convergence.py` | Statistical error vs cost: Monte Carlo vs IAE |
| `scripts/run_bias.py` | Discretisation + payoff-encoding bias vs qubit count |
| `scripts/run_noise.py` | How depolarising noise erodes Grover amplification on hardware-like circuits |

## Limits, stated plainly
- The price-history model assumes the future's shocks look like the past's. A first-of-its-kind event is not in the sample.
- The bell curve ignores fat tails. The option-based market distribution captures them, but it is risk-neutral (it prices risk;
  it does not measure real-world odds), and it is only as good as the option quotes behind it.
- The quantum speed-up is in **calls to A**, not wall-clock time. Simulating the circuit classically is far slower than Monte Carlo.
- On real hardware, noise destroys the deep Grover circuits long before any advantage appears (`run_noise.py`).

## Layout
```
app.py                 Streamlit website
.streamlit/config.toml colours and font
  i18n.py              Korean / English text
  web_charts.py        interactive Plotly charts with explanatory hovers
run_dashboard.py       launcher (main-thread Qiskit import)
src/qstock/
  market.py            Yahoo loaders, implied vol solver, demo market
  stockdist.py         price-history forecast (filtered historical simulation)
  implied.py           option-implied (Breeden–Litzenberger) distribution, advanced
  ibm.py               IBM hardware runs, noisy preview, readout correction, job log
  forecast.py          forecast distribution, cell-averaged encoding, quantum estimators
  quantum.py           Qiskit circuits and amplitude estimation
  backtest.py          walk-forward calibration
  classical.py         Black–Scholes, Monte Carlo, point-sampled grid
  charts.py            figures shared by the CLI and dashboard
scripts/               reports and experiments
tests/                 pytest suite (no network needed)
```
