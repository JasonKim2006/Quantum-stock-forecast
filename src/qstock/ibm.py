"""Run the forecast circuit on IBM quantum hardware (or a free, noisy local preview).

What is sent
------------
For a probability a = P(objective qubit = 1) encoded by the state-preparation
circuit A (see :func:`qstock.quantum.build_exact_problem`):

* ``k=0``: A, then measure the objective qubit. The 1-rate estimates a directly.
* ``k=1``: A followed by one Grover iteration Q. Ideally the 1-rate is
  sin^2(3 theta) with a = sin^2(theta); this is the step amplitude estimation
  relies on, and the one hardware noise hurts most because Q roughly triples
  the circuit depth.
* Two readout-calibration circuits (prepare |0> and |1> on the same physical
  qubit) to correct measurement error: with e0 = P(read 1 | 0) and
  e1 = P(read 0 | 1),  p_true = (p_measured - e0) / (1 - e0 - e1).

Full iterative amplitude estimation needs many rounds with deeper circuits, which
the free plan's 10 minutes per 28 days can't cover and which today's noise would
wash out anyway. Comparing k=0 and k=1 with the exact values shows *where* that
breaks down, and is a meaningful hardware experiment in its own right.

Credentials
-----------
Nothing here reads or stores an API key. ``QiskitRuntimeService()`` uses the
account saved once with ``QiskitRuntimeService.save_account`` in ~/.qiskit.
Submitted job IDs and the metadata needed to interpret them are kept in
results/ibm_jobs.json, so results can be fetched later (queues can be long).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from .quantum import build_exact_problem

JOBS_FILE = Path(__file__).resolve().parents[2] / "results" / "ibm_jobs.json"
TWO_Q = ("cz", "ecr", "cx")


@dataclass
class HardwarePlan:
    """Circuits ready to run plus everything needed to interpret the counts later."""

    circuits: list = field(repr=False)
    labels: list[str]
    exact: dict[str, float]       # ideal P(objective = 1) per label
    scale: float                  # post-processing: estimate = scale * a
    two_qubit_gates: dict[str, int]
    depth: dict[str, int]
    physical_qubit: int
    backend: str


def _measured(circuit: QuantumCircuit, objective: int) -> QuantumCircuit:
    qc = circuit.copy()
    qc.add_register(ClassicalRegister(1, "c"))
    qc.measure(objective, 0)
    return qc


def plan_circuits(backend, pmf, cond, include_grover: bool = True, optimization_level: int = 3) -> HardwarePlan:
    """Build and transpile the k=0 (and k=1) circuits plus readout calibration for ``backend``."""
    problem = build_exact_problem(pmf, cond)
    A = problem.state_preparation
    n = A.num_qubits - 1
    a = float(np.dot(pmf / pmf.sum(), np.clip(cond / (cond.max() if cond.max() > 0 else 1), 0, 1)))
    theta = np.arcsin(np.sqrt(a))
    scale = float(cond.max()) if cond.max() > 0 else 1.0

    logical = {"k=0": _measured(A, n)}
    exact = {"k=0": a}
    if include_grover:
        logical["k=1"] = _measured(A.compose(problem.grover_operator), n)
        exact["k=1"] = float(np.sin(3 * theta) ** 2)

    pm = generate_preset_pass_manager(backend=backend, optimization_level=optimization_level, seed_transpiler=7)
    isa = {name: pm.run(qc) for name, qc in logical.items()}
    phys = int(isa["k=0"].layout.final_index_layout()[n])

    # Readout calibration on the same physical qubit the objective lands on.
    cal_pm = generate_preset_pass_manager(backend=backend, optimization_level=1, initial_layout=[phys])
    for label, flip in (("cal0", False), ("cal1", True)):
        qc = QuantumCircuit(1, 1)
        if flip:
            qc.x(0)
        qc.measure(0, 0)
        isa[label] = cal_pm.run(qc)

    def two_q(qc):
        ops = qc.count_ops()
        return int(sum(ops.get(g, 0) for g in TWO_Q))

    labels = list(isa)
    return HardwarePlan(
        circuits=[isa[k] for k in labels],
        labels=labels,
        exact=exact,
        scale=scale,
        two_qubit_gates={k: two_q(isa[k]) for k in logical},
        depth={k: int(isa[k].depth()) for k in logical},
        physical_qubit=phys,
        backend=getattr(backend, "name", str(backend)),
    )


def _ones_rate(pub_result) -> tuple[float, int]:
    data = pub_result.data
    reg = getattr(data, "c", None) or next(iter(data.values()))
    counts = reg.get_counts()
    shots = sum(counts.values())
    return counts.get("1", 0) / shots, shots


def analyse(labels: list[str], pub_results, exact: dict[str, float], scale: float) -> dict:
    """Raw and readout-corrected 1-rates, with the exact values for comparison."""
    rates = {lab: _ones_rate(r) for lab, r in zip(labels, pub_results)}
    e0 = rates["cal0"][0]            # read 1 when |0> was prepared
    e1 = 1 - rates["cal1"][0]        # read 0 when |1> was prepared
    denom = max(1 - e0 - e1, 1e-6)
    out = {"readout_error_0to1": e0, "readout_error_1to0": e1, "rows": []}
    for lab in exact:
        raw, shots = rates[lab]
        fixed = float(np.clip((raw - e0) / denom, 0, 1))
        se = float(np.sqrt(max(raw * (1 - raw), 1e-12) / shots))
        out["rows"].append(
            {
                "circuit": lab,
                "exact": exact[lab],
                "raw": raw,
                "corrected": fixed,
                "stderr": se,
                "shots": shots,
            }
        )
    # The k=0 rate estimates a directly, so it maps to the quantity we were after.
    k0 = next(r for r in out["rows"] if r["circuit"] == "k=0")
    out["estimate_raw"] = scale * k0["raw"]
    out["estimate_corrected"] = scale * k0["corrected"]
    out["estimate_exact"] = scale * k0["exact"]
    return out


# ------------------------------------------------------------------ free preview (local, noisy)
def preview(pmf, cond, fake_name: str = "FakeFez", shots: int = 4000, include_grover: bool = True, seed: int = 1):
    """Run the exact same circuits on a local simulator that uses an IBM device's calibration data.

    Costs no IBM time. The noise model comes from a snapshot of that device's error
    rates, so it is a realistic but not exact stand-in for the live machine.
    """
    from qiskit_ibm_runtime import SamplerV2
    from qiskit_ibm_runtime import fake_provider

    backend = getattr(fake_provider, fake_name)()
    plan = plan_circuits(backend, pmf, cond, include_grover)
    sampler = SamplerV2(mode=backend, options={"simulator": {"seed_simulator": seed}})
    result = sampler.run(plan.circuits, shots=shots).result()
    return plan, analyse(plan.labels, result, plan.exact, plan.scale)


# ------------------------------------------------------------------ real hardware
def connect():
    """Saved IBM account, or raise with a readable message."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    return QiskitRuntimeService()


def remaining_minutes(service) -> float | None:
    try:
        u = service.usage()
    except Exception:
        return None
    if "usage_remaining_ru" in u:
        return float(u["usage_remaining_ru"])  # 1 RU is about 1 QPU-minute on the Open Plan
    if "usage_remaining_seconds" in u:
        return float(u["usage_remaining_seconds"]) / 60
    return None


def submit(service, pmf, cond, shots: int = 4000, include_grover: bool = True, backend_name: str | None = None,
           meta: dict | None = None) -> dict:
    """Transpile for a real device, submit one Sampler job, and record it in JOBS_FILE."""
    from qiskit_ibm_runtime import SamplerV2

    backend = service.backend(backend_name) if backend_name else service.least_busy(operational=True, simulator=False)
    plan = plan_circuits(backend, pmf, cond, include_grover)
    job = SamplerV2(mode=backend).run(plan.circuits, shots=shots)
    record = {
        "job_id": job.job_id(),
        "backend": plan.backend,
        "submitted": time.strftime("%Y-%m-%d %H:%M"),
        "labels": plan.labels,
        "exact": plan.exact,
        "scale": plan.scale,
        "two_qubit_gates": plan.two_qubit_gates,
        "depth": plan.depth,
        "physical_qubit": plan.physical_qubit,
        "shots": shots,
        **(meta or {}),
    }
    jobs = load_jobs()
    jobs.insert(0, record)
    save_jobs(jobs)
    return record


def fetch(service, record: dict) -> tuple[str, dict | None, float | None]:
    """(status, analysis or None, quantum seconds used or None) for a recorded job."""
    job = service.job(record["job_id"])
    status = job.status()
    status = status if isinstance(status, str) else getattr(status, "name", str(status))
    if status != "DONE":
        return status, None, None
    res = job.result()
    used = None
    try:
        used = float(job.usage())
    except Exception:
        pass
    return status, analyse(record["labels"], res, record["exact"], record["scale"]), used


def load_jobs() -> list[dict]:
    try:
        return json.loads(JOBS_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_jobs(jobs: list[dict]) -> None:
    JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
    JOBS_FILE.write_text(json.dumps(jobs, indent=2))


def plan_summary(plan: HardwarePlan) -> dict:
    d = asdict(plan)
    d.pop("circuits")
    return d
