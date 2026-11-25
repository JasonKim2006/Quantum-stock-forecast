"""Experiment 2: where does the *systematic* error come from?

An ideal quantum computer still prices a slightly different model than
Black-Scholes, because of two approximations:

  1. Discretisation + truncation: S_T is loaded on a grid of 2**n points
     spanning mean +/- 3 std.  -> |discretised - Black-Scholes|
  2. Payoff encoding: the payoff is mapped to a rotation angle and the
     sin(x) ~ x approximation is controlled by c_approx.
                                 -> |circuit - discretised|

Both are computed exactly (statevector), so no sampling noise is involved.

Usage:  python scripts/run_bias.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from qstock import OptionParams, black_scholes_call, discretised_call_price
from qstock.plotting import INK_2, MARKERS, SERIES, apply_style
from qstock.quantum import build_problem, ideal_circuit_price

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

N_QUBITS = [2, 3, 4, 5, 6, 7]
C_APPROX = [0.025, 0.05, 0.1, 0.25]
N_STD = [3, 4, 5, 6]


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    p = OptionParams()
    bs = black_scholes_call(p)

    rows = []
    for n in N_QUBITS:
        disc = discretised_call_price(p, n)
        row = dict(n_qubits=n, circuit_qubits=None, black_scholes=bs, discretised=disc,
                   discretisation_error=abs(disc - bs))
        for c in C_APPROX:
            problem = build_problem(p, n, c_approx=c)
            row["circuit_qubits"] = problem.state_preparation.num_qubits
            circ = ideal_circuit_price(problem, p)
            row[f"circuit_c{c}"] = circ
            row[f"encoding_error_c{c}"] = abs(circ - disc)
            row[f"total_error_c{c}"] = abs(circ - bs)
        rows.append(row)
        print(f"n={n}  disc err={row['discretisation_error']:.2e}  "
              + "  ".join(f"enc(c={c})={row[f'encoding_error_c{c}']:.2e}" for c in C_APPROX),
              flush=True)

    with open(RESULTS / "bias.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    import matplotlib.pyplot as plt

    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # Left panel: discretisation + truncation, for several truncation widths.
    # This is a purely classical computation, so we can afford larger n.
    trunc_rows = []
    n_wide = np.arange(2, 11)
    for i, k in enumerate(N_STD):
        pk = OptionParams(n_std=k)
        bs_k = black_scholes_call(pk)
        err = np.array([abs(discretised_call_price(pk, int(n)) - bs_k) for n in n_wide])
        trunc_rows += [dict(n_std=k, n_qubits=int(n), discretisation_error=float(e))
                       for n, e in zip(n_wide, err)]
        ax1.semilogy(n_wide, err, MARKERS[i] + "-", color=SERIES[i], label=f"±{k}σ window")
        ax1.annotate(f"±{k}σ", (n_wide[-1], err[-1]), textcoords="offset points",
                     xytext=(6, -3), color=INK_2, fontsize=8.5)
    with open(RESULTS / "truncation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trunc_rows[0]))
        w.writeheader()
        w.writerows(trunc_rows)
    ax1.set_xlabel("Qubits in price register n  (grid of 2ⁿ points)")
    ax1.set_ylabel("|discretised − Black-Scholes|")
    ax1.set_title("Discretisation error plateaus at truncation error")
    ax1.set_xticks(n_wide)
    ax1.set_xlim(n_wide[0] - 0.3, n_wide[-1] + 1.3)
    ax1.legend(loc="lower left", ncols=2)

    ns = np.array(N_QUBITS)

    for i, c in enumerate(C_APPROX):
        err = np.array([r[f"encoding_error_c{c}"] for r in rows])
        ax2.semilogy(ns, err, MARKERS[i] + "-", color=SERIES[i], label=f"c = {c}")
        ax2.annotate(f"c = {c}", (ns[-1], err[-1]), textcoords="offset points",
                     xytext=(6, -3), color=INK_2, fontsize=8.5)
    ax2.set_xlabel("Qubits in price register n")
    ax2.set_ylabel("|circuit − discretised|")
    ax2.set_title("Encoding error scales as c²")
    ax2.set_xticks(ns)
    ax2.set_xlim(ns[0] - 0.3, ns[-1] + 1.1)
    ax2.legend(loc="upper left", bbox_to_anchor=(0.0, 0.86), ncols=4, handlelength=1.5,
               columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(FIGURES / "bias.png")
    print("saved", FIGURES / "bias.png")


if __name__ == "__main__":
    main()
