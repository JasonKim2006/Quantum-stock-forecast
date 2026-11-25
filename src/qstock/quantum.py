"""Quantum amplitude estimation of the European call price with Qiskit.

Pipeline
--------
1. ``LogNormalDistribution`` loads the discretised S_T distribution into an
   n-qubit register: sum_i sqrt(p_i) |i>.
2. ``LinearAmplitudeFunctionGate`` rotates an objective qubit so that
   P(objective = 1) is an affine function of the payoff max(S_T - K, 0).
3. Iterative Amplitude Estimation (IAE) estimates that probability, which is
   then mapped back to a price.

Cost accounting
---------------
``qiskit_algorithms`` reports ``num_oracle_queries = shots * sum(k)``, which
counts applications of the Grover operator Q only. Q = A S_0 A^dagger S_f uses
the state-preparation circuit A twice, and every shot also runs A once, so one
shot at Grover power k costs ``2k + 1`` calls to A. A call to A is the quantum
analogue of drawing one Monte Carlo sample, so ``a_calls`` is the number to
compare against the classical sample count.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    LinearAmplitudeFunctionGate,
    StatePreparation,
    UCRYGate,
    UnitaryGate,
)
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Operator, Statevector
from qiskit_algorithms import EstimationProblem, IterativeAmplitudeEstimation
from qiskit_finance.circuit.library import LogNormalDistribution

from .params import OptionParams

DEFAULT_C_APPROX = 0.25


@dataclass(frozen=True)
class QuantumEstimate:
    price: float
    ci_low: float
    ci_high: float
    a_calls: int
    grover_queries: int
    rounds: int
    shots: int


def build_problem(
    p: OptionParams, n_qubits: int, c_approx: float = DEFAULT_C_APPROX
) -> EstimationProblem:
    """Assemble the amplitude-estimation problem for the call option.

    The returned problem's ``post_processing`` maps the estimated amplitude to
    the *undiscounted* expected payoff; multiply by ``p.discount`` for a price.

    Args:
        p: Option parameters.
        n_qubits: Qubits in the uncertainty register (grid of 2**n_qubits points).
        c_approx: Rescaling factor of the payoff encoding. Smaller values make
            the sin(x) ~ x approximation more accurate at the cost of a smaller
            signal (and therefore more samples for the same price accuracy).
    """
    low, high = p.bounds
    dist = LogNormalDistribution(n_qubits, mu=p.mu, sigma=p.sigma**2, bounds=(low, high))
    payoff = LinearAmplitudeFunctionGate(
        n_qubits,
        slope=[0, 1],
        offset=[0, 0],
        domain=(low, high),
        image=(0, high - p.K),
        rescaling_factor=c_approx,
        breakpoints=[low, p.K],
    )
    circuit = QuantumCircuit(payoff.num_qubits, name="A")
    circuit.append(dist, range(n_qubits))
    circuit.append(payoff, range(payoff.num_qubits))
    return EstimationProblem(
        state_preparation=circuit,
        objective_qubits=[n_qubits],
        post_processing=payoff.post_processing,
    )


def emulated_problem(problem: EstimationProblem) -> EstimationProblem:
    """Return an equivalent problem whose A and Q are single dense unitaries.

    Simulating ``Q^k A|0>`` gate by gate re-decomposes the (deep) Grover
    operator k times, which dominates run time for small epsilon. For the 5-8
    qubit circuits used here, A and Q fit comfortably in memory as dense
    matrices, so we precompute them once. The output distribution is identical
    to the gate-level circuit (checked in ``tests/test_quantum.py``); this is an
    exact, noiseless emulation and is *not* used for the noise study.
    """
    n = problem.state_preparation.num_qubits
    a_gate = UnitaryGate(Operator(problem.state_preparation), label="A")
    q_gate = UnitaryGate(Operator(problem.grover_operator), label="Q")
    a_circ = QuantumCircuit(n, name="A")
    a_circ.append(a_gate, range(n))
    q_circ = QuantumCircuit(n, name="Q")
    q_circ.append(q_gate, range(n))
    return EstimationProblem(
        state_preparation=a_circ,
        grover_operator=q_circ,
        objective_qubits=problem.objective_qubits,
        post_processing=problem.post_processing,
    )


def good_state_probability(problem: EstimationProblem, k: int = 0) -> float:
    """Exact probability that the objective qubit reads 1 after Q^k A|0>."""
    circuit = problem.state_preparation.compose(problem.grover_operator.power(k))
    return float(Statevector(circuit).probabilities(problem.objective_qubits)[1])


def ideal_circuit_price(problem: EstimationProblem, p: OptionParams) -> float:
    """Price encoded in the circuit, computed exactly with a statevector.

    Differs from :func:`qstock.classical.discretised_call_price` only through the
    first-order approximation used when encoding the payoff.
    """
    return float(p.discount * problem.post_processing(good_state_probability(problem)))


def estimate_price_iae(
    problem: EstimationProblem,
    p: OptionParams,
    epsilon: float,
    alpha: float = 0.05,
    shots: int = 100,
    seed: int | None = None,
) -> QuantumEstimate:
    """Run Iterative Amplitude Estimation and convert the result to a price.

    Args:
        epsilon: Target half-width of the confidence interval on the amplitude.
        alpha: Confidence level is ``1 - alpha``.
        shots: Shots per IAE round.
        seed: Seed for the sampler's measurement randomness.
    """
    sampler = StatevectorSampler(default_shots=shots, seed=seed)
    iae = IterativeAmplitudeEstimation(epsilon_target=epsilon, alpha=alpha, sampler=sampler)
    result = iae.estimate(problem)

    powers = list(result.powers[1:])  # powers[0] is a placeholder, one entry per round
    a_calls = int(shots * sum(2 * k + 1 for k in powers))
    ci_low, ci_high = result.confidence_interval_processed
    return QuantumEstimate(
        price=float(p.discount * result.estimation_processed),
        ci_low=float(p.discount * ci_low),
        ci_high=float(p.discount * ci_high),
        a_calls=a_calls,
        grover_queries=int(result.num_oracle_queries),
        rounds=len(powers),
        shots=shots,
    )


# --------------------------------------------------------------------------
# Exact encoding of an arbitrary payoff (used by the market / forecast layer)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ExpectationEstimate:
    """IAE estimate of E[f(S_T)] (undiscounted) with its confidence interval."""

    value: float
    ci_low: float
    ci_high: float
    a_calls: int
    exact: float  # value the circuit encodes, from a statevector


def build_exact_problem(pmf: np.ndarray, payoff: np.ndarray) -> EstimationProblem:
    """A|0> = sum_i sqrt(p_i)|i> (sqrt(1-g_i)|0> + sqrt(g_i)|1>), g_i = f_i / max f.

    Unlike :func:`build_problem`, which approximates a piecewise-linear payoff
    with ``LinearAmplitudeFunctionGate``, this uses one uniformly controlled
    RY rotation per grid point, so P(objective = 1) = E[f] / max f *exactly*.
    The cost is 2**n controlled rotations, which is fine for the 3-6 qubit
    registers used here and lets one circuit handle calls, puts and digital
    ("price ends above X") payoffs alike.
    """
    pmf = np.asarray(pmf, float)
    payoff = np.asarray(payoff, float)
    n = int(np.log2(len(pmf)))
    if 2**n != len(pmf) or len(payoff) != len(pmf):
        raise ValueError("pmf and payoff must have the same power-of-two length.")
    if np.any(payoff < 0):
        raise ValueError("Payoff must be non-negative.")
    scale = float(payoff.max()) if payoff.max() > 0 else 1.0
    angles = 2 * np.arcsin(np.sqrt(np.clip(payoff / scale, 0.0, 1.0)))

    circuit = QuantumCircuit(n + 1, name="A")
    circuit.append(StatePreparation(np.sqrt(pmf / pmf.sum())), range(n))
    # UCRYGate: qubit 0 is the target, the rest are controls (little-endian index).
    circuit.append(UCRYGate(list(angles)), [n, *range(n)])
    return EstimationProblem(
        state_preparation=circuit,
        objective_qubits=[n],
        post_processing=lambda a: scale * a,
    )


def estimate_expectation(
    problem: EstimationProblem,
    epsilon: float = 0.005,
    alpha: float = 0.05,
    shots: int = 100,
    seed: int | None = None,
) -> ExpectationEstimate:
    """Iterative Amplitude Estimation of E[f] for a problem from build_exact_problem."""
    fast = emulated_problem(problem)
    sampler = StatevectorSampler(default_shots=shots, seed=seed)
    iae = IterativeAmplitudeEstimation(epsilon_target=epsilon, alpha=alpha, sampler=sampler)
    result = iae.estimate(fast)
    powers = list(result.powers[1:])
    lo, hi = result.confidence_interval_processed
    return ExpectationEstimate(
        value=float(result.estimation_processed),
        ci_low=float(lo),
        ci_high=float(hi),
        a_calls=int(shots * sum(2 * k + 1 for k in powers)),
        exact=float(problem.post_processing(good_state_probability(problem))),
    )
