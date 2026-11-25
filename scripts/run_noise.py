"""Experiment 3: how hardware noise erodes the quantum speed-up.

Amplitude estimation gets its 1/N scaling by running Q^k A|0> for growing
Grover powers k. Each power multiplies circuit depth, so under gate noise the
measured probability decays towards 1/2 (the maximally mixed value) and the
information that high-k rounds were supposed to contribute is lost.

For each two-qubit depolarizing error rate we:
  * transpile Q^k A to a hardware-like basis {cx, rz, sx, x},
  * simulate the full density matrix with Aer (exact, no shot noise),
  * compare P(objective = 1) to the ideal sin^2((2k+1) theta).

Single-qubit gates get 1/10 of the two-qubit rate, a common rule of thumb.

Usage:  python scripts/run_noise.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from qstock import OptionParams
from qstock.plotting import INK_2, MARKERS, SERIES, apply_style
from qstock.quantum import build_problem, good_state_probability

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

BASIS = ["cx", "rz", "sx", "x"]
ERROR_RATES = [1e-4, 1e-3, 1e-2]  # two-qubit depolarizing probability
K_VALUES = list(range(9))


def noise_model(p2: float) -> NoiseModel:
    nm = NoiseModel(basis_gates=BASIS)
    nm.add_all_qubit_quantum_error(depolarizing_error(p2 / 10, 1), ["rz", "sx", "x"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    return nm


def compiled_circuit(problem, k: int) -> QuantumCircuit:
    qc = problem.state_preparation.compose(problem.grover_operator.power(k))
    qc = transpile(qc, basis_gates=BASIS, optimization_level=1, seed_transpiler=7)
    return qc


def noisy_probability(qc: QuantumCircuit, objective: int, p2: float) -> float:
    qc = qc.copy()
    qc.save_probabilities([objective], label="p")
    sim = AerSimulator(method="density_matrix", noise_model=noise_model(p2))
    data = sim.run(qc, shots=1).result().data()["p"]
    return float(np.asarray(data)[1])


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    p = OptionParams()
    problem = build_problem(p, n_qubits=3)
    obj = problem.objective_qubits[0]
    a = good_state_probability(problem)
    theta = np.arcsin(np.sqrt(a))

    rows = []
    for k in K_VALUES:
        qc = compiled_circuit(problem, k)
        ops = qc.count_ops()
        ideal = float(np.sin((2 * k + 1) * theta) ** 2)
        row = dict(k=k, a_calls_per_shot=2 * k + 1, cx_count=int(ops.get("cx", 0)),
                   depth=qc.depth(), ideal=ideal)
        for p2 in ERROR_RATES:
            prob = noisy_probability(qc, obj, p2)
            row[f"p_noisy_{p2:g}"] = prob
            # Fraction of the ideal signal (distance from 1/2) that survives.
            row[f"visibility_{p2:g}"] = (prob - 0.5) / (ideal - 0.5) if abs(ideal - 0.5) > 1e-3 else np.nan
        rows.append(row)
        print(f"k={k}  CX={row['cx_count']:>5}  depth={row['depth']:>5}  ideal={ideal:.3f}  "
              + "  ".join(f"p2={p2:g}: {row[f'p_noisy_{p2:g}']:.3f}" for p2 in ERROR_RATES),
              flush=True)

    with open(RESULTS / "noise.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ks = np.array([r["k"] for r in rows])
    kk = np.linspace(0, ks.max(), 400)

    ax1.plot(kk, np.sin((2 * kk + 1) * theta) ** 2, "-", color=SERIES[0], lw=1.2, alpha=0.45)
    ax1.plot(ks, [r["ideal"] for r in rows], MARKERS[0], color=SERIES[0])
    for i, p2 in enumerate(ERROR_RATES, start=1):
        ax1.plot(ks, [r[f"p_noisy_{p2:g}"] for r in rows], MARKERS[i] + "-", color=SERIES[i],
                 lw=1.2)
    ax1.axhline(0.5, color=INK_2, lw=0.8, ls="--")
    handles = [Line2D([], [], color=SERIES[0], marker=MARKERS[0], lw=1.2, alpha=0.8, label="Ideal")]
    handles += [Line2D([], [], color=SERIES[i], marker=MARKERS[i], lw=1.2, label=f"CX error {p2:g}")
                for i, p2 in enumerate(ERROR_RATES, start=1)]
    handles.append(Line2D([], [], color=INK_2, lw=0.8, ls="--", label="Fully mixed (P = 0.5)"))
    ax1.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=3)
    ax1.set_xlabel("Grover power k")
    ax1.set_ylabel("P(objective qubit = 1)")
    ax1.set_title("Noise flattens the Grover oscillation")
    ax1.set_ylim(-0.03, 1.05)
    ax1.set_xticks(ks)

    # Right: surviving signal. Only points where the ideal value is far from 1/2,
    # otherwise the ratio is dominated by round-off.
    for i, p2 in enumerate(ERROR_RATES, start=1):
        pts = [(r["k"], r[f"visibility_{p2:g}"]) for r in rows
               if abs(r["ideal"] - 0.5) > 0.2 and r[f"visibility_{p2:g}"] > 1e-3]
        if not pts:
            continue
        x, y = zip(*pts)
        ax2.semilogy(x, y, MARKERS[i] + "-", color=SERIES[i], lw=1.2, label=f"CX error {p2:g}")
        ax2.annotate(f"{p2:g}", (x[-1], y[-1]), textcoords="offset points", xytext=(6, -3),
                     color=INK_2, fontsize=8.5)
    ax2.set_xlabel("Grover power k")
    ax2.set_ylabel("Surviving signal  (P − ½) / (P_ideal − ½)")
    ax2.set_title("Signal decays exponentially with depth")
    ax2.set_xticks(ks)
    ax2.set_xlim(-0.3, ks.max() + 0.9)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=3)
    fig.tight_layout()
    fig.savefig(FIGURES / "noise.png")
    print("saved", FIGURES / "noise.png")


if __name__ == "__main__":
    main()
