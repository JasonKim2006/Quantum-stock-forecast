"""Experiment 1: statistical error vs cost, classical Monte Carlo vs IAE.

Cost is measured in calls to the "sampling primitive":
  * Monte Carlo: one draw of S_T.
  * Quantum:     one application of the state-preparation circuit A
                 (each shot at Grover power k costs 2k + 1 calls).

Each method is scored against the value it is an unbiased estimator of
(Black-Scholes for MC, the exact circuit price for IAE), so this isolates the
*sampling* error. Model bias is studied separately in run_bias.py.

Usage:  python scripts/run_convergence.py [--seeds 30] [--quick]
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from qstock import OptionParams, black_scholes_call, mc_call_price
from qstock.plotting import INK_2, MARKERS, SERIES, apply_style
from qstock.quantum import build_problem, emulated_problem, estimate_price_iae, ideal_circuit_price

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"


def fit_slope(cost: np.ndarray, err: np.ndarray) -> tuple[float, float]:
    """Least-squares fit of log(err) = slope * log(cost) + log(c)."""
    slope, intercept = np.polyfit(np.log10(cost), np.log10(err), 1)
    return float(slope), float(10**intercept)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--n-qubits", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="small sweep for a smoke test")
    args = ap.parse_args()

    p = OptionParams()
    bs = black_scholes_call(p)
    problem = build_problem(p, args.n_qubits)
    target_q = ideal_circuit_price(problem, p)
    fast = emulated_problem(problem)

    seeds = 5 if args.quick else args.seeds
    eps_grid = np.logspace(-2, -3, 3) if args.quick else np.logspace(-2, -5, 10)
    mc_grid = np.logspace(2, 4, 3) if args.quick else np.logspace(2, 7, 11)

    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    # ---- Quantum: Iterative Amplitude Estimation -------------------------
    q_rows = []
    t0 = time.time()
    for eps in eps_grid:
        runs = [estimate_price_iae(fast, p, epsilon=eps, seed=s) for s in range(seeds)]
        errs = np.array([r.price - target_q for r in runs])
        cost = np.array([r.a_calls for r in runs])
        covered = np.mean([r.ci_low <= target_q <= r.ci_high for r in runs])
        q_rows.append(
            dict(
                epsilon=eps,
                mean_a_calls=cost.mean(),
                rms_error=float(np.sqrt(np.mean(errs**2))),
                mean_abs_error=float(np.mean(np.abs(errs))),
                ci_coverage=float(covered),
            )
        )
        print(
            f"IAE eps={eps:.1e}  A-calls={cost.mean():>10.0f}  "
            f"RMS={q_rows[-1]['rms_error']:.2e}  coverage={covered:.2f}  "
            f"[{time.time() - t0:.0f}s]",
            flush=True,
        )

    # ---- Classical: Monte Carlo ------------------------------------------
    rng = np.random.default_rng(12345)
    mc_rows = []
    mc_trials = 20 if args.quick else 200
    for n in mc_grid.astype(int):
        trials = mc_trials if n <= 10**6 else max(20, mc_trials // 10)
        errs = np.array([mc_call_price(p, int(n), rng) - bs for _ in range(trials)])
        mc_rows.append(dict(samples=int(n), rms_error=float(np.sqrt(np.mean(errs**2)))))
        print(f"MC  N={n:>9d}  RMS={mc_rows[-1]['rms_error']:.2e}", flush=True)

    # ---- Fits and crossover -----------------------------------------------
    q_cost = np.array([r["mean_a_calls"] for r in q_rows])
    q_err = np.array([r["rms_error"] for r in q_rows])
    mc_cost = np.array([r["samples"] for r in mc_rows], dtype=float)
    mc_err = np.array([r["rms_error"] for r in mc_rows])
    q_slope, q_c = fit_slope(q_cost, q_err)
    mc_slope, mc_c = fit_slope(mc_cost, mc_err)
    # cost at which the two fitted lines meet
    crossover = (q_c / mc_c) ** (1 / (mc_slope - q_slope))

    with open(RESULTS / "convergence_quantum.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(q_rows[0]))
        w.writeheader()
        w.writerows(q_rows)
    with open(RESULTS / "convergence_classical.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mc_rows[0]))
        w.writeheader()
        w.writerows(mc_rows)
    with open(RESULTS / "convergence_summary.txt", "w") as f:
        f.write(f"Black-Scholes price         : {bs:.6f}\n")
        f.write(f"Circuit target ({args.n_qubits} qubits)  : {target_q:.6f}\n")
        f.write(f"Seeds per IAE point         : {seeds}\n")
        f.write(f"MC fitted slope             : {mc_slope:.3f}  (theory -0.5)\n")
        f.write(f"IAE fitted slope            : {q_slope:.3f}  (theory -1.0)\n")
        f.write(f"Fitted-line crossover cost  : {crossover:.3g} calls\n")
    print((RESULTS / "convergence_summary.txt").read_text())

    # ---- Figure -------------------------------------------------------------
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    xs = np.logspace(np.log10(min(q_cost.min(), mc_cost.min())), np.log10(max(q_cost.max(), mc_cost.max())), 50)
    ax.loglog(mc_cost, mc_err, MARKERS[0] + "-", color=SERIES[0], label="Classical Monte Carlo")
    ax.loglog(q_cost, q_err, MARKERS[1] + "-", color=SERIES[1], label="Quantum (IAE, noiseless)")
    ax.loglog(xs, mc_c * xs**mc_slope, ":", color=SERIES[0], lw=1.2)
    ax.loglog(xs, q_c * xs**q_slope, ":", color=SERIES[1], lw=1.2)
    ax.annotate(f"slope {mc_slope:.2f}", (mc_cost[-1], mc_err[-1]), textcoords="offset points",
                xytext=(-8, 10), ha="right", color=INK_2, fontsize=9)
    ax.annotate(f"slope {q_slope:.2f}", (q_cost[-1], q_err[-1]), textcoords="offset points",
                xytext=(-8, -16), ha="right", color=INK_2, fontsize=9)
    if xs.min() < crossover < xs.max():
        ax.axvline(crossover, color=INK_2, lw=0.8, ls="--")
        ax.annotate(f"crossover ≈ {crossover:.1e} calls", (crossover, ax.get_ylim()[1]),
                    textcoords="offset points", xytext=(6, -14), color=INK_2, fontsize=9)
    ax.set_xlabel("Cost: samples (MC) or calls to state-preparation circuit A (quantum)")
    ax.set_ylabel("RMS price error")
    ax.set_title("Sampling error vs cost: quadratic speed-up in the noiseless limit")
    ax.legend(loc="lower left")
    fig.savefig(FIGURES / "convergence.png")
    print("saved", FIGURES / "convergence.png")


if __name__ == "__main__":
    main()
