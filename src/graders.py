"""Physics‑based graders for quantum algorithm tasks.

Each grader receives ``circuit_gates`` (a list of ``{"name": str, "qubits": List[int]}``),
the number of qubits in the environment, and a ``problem_params`` dictionary.
It returns a ``float`` in the range ``[0.0, 1.0]`` representing how well the
provided circuit solves the task.  The implementation uses Qiskit state‑vector
simulation to compute a physically meaningful score rather than a simple
structural heuristic.
"""

from typing import Any, Dict, List
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.circuit.library import QFT

# ---------------------------------------------------------------------------
# Helper: map gate names to Qiskit methods (fixed‑angle rotations where needed).
# ---------------------------------------------------------------------------
_GATE_MAP = {
    "h":  lambda c, q: c.h(q[0]),
    "cx": lambda c, q: c.cx(q[0], q[1]),
    "t":  lambda c, q: c.t(q[0]),
    "s":  lambda c, q: c.s(q[0]),
    "x":  lambda c, q: c.x(q[0]),
    "ry": lambda c, q: c.ry(np.pi / 4, q[0]),
    "rz": lambda c, q: c.rz(np.pi / 4, q[0]),
    "rx": lambda c, q: c.rx(np.pi / 4, q[0]),
}

# ---------------------------------------------------------------------------
# H₂ Hamiltonian (STO‑3G, Jordan‑Wigner, 2‑qubit) for the VQE grader.
# ---------------------------------------------------------------------------
_H2_HAMILTONIAN = SparsePauliOp.from_list(
    [
        ("II", -0.8105),
        ("ZI", 0.1722),
        ("IZ", -0.2228),
        ("ZZ", 0.1208),
        ("YY", -0.0453),
        ("XX", 0.1722),
    ]
)
_H2_GROUND_ENERGY = -1.137  # Hartree


def _gates_to_circuit(gates: List[Dict[str, Any]], n_qubits: int) -> QuantumCircuit:
    """Recreate a :class:`QuantumCircuit` from a list of gate dictionaries.

    Invalid or out‑of‑range gates are ignored.  This function is used by all
    graders to obtain a concrete circuit for simulation.
    """
    circ = QuantumCircuit(n_qubits)
    for gate in gates:
        name = gate.get("name", "")
        qubits = gate.get("qubits", [])
        if name in _GATE_MAP and qubits and all(q < n_qubits for q in qubits):
            try:
                _GATE_MAP[name](circ, qubits)
            except Exception:
                # Silently skip malformed specifications.
                pass
    return circ


def grade_parity_optimization(
    circuit_gates: List[Dict[str, Any]],
    qubit_count: int,
    problem_params: Dict[str, Any],
) -> float:
    """Score based on fidelity with the ideal 3‑qubit parity circuit.

    The ideal circuit applies ``CX(0, 2)`` then ``CX(1, 2)``.  The score is the
    squared overlap between the ideal statevector and the state produced by the
    submitted gates.
    """
    if not circuit_gates:
        return 0.0
    n = max(qubit_count, 3)
    ideal = QuantumCircuit(n)
    ideal.cx(0, 2)
    ideal.cx(1, 2)
    ideal_sv = Statevector(ideal)
    circ = _gates_to_circuit(circuit_gates, n)
    try:
        overlap = abs(complex(ideal_sv.inner(Statevector(circ)))) ** 2
        return float(np.clip(overlap, 0.0, 1.0))
    except Exception:
        return 0.0


def grade_shors_factoring(
    circuit_gates: List[Dict[str, Any]],
    qubit_count: int,
    problem_params: Dict[str, Any],
) -> float:
    """Score based on fidelity with a 3‑qubit QFT (no final swaps)."""
    if not circuit_gates:
        return 0.0
    n = max(qubit_count, 3)
    ideal_sv = Statevector(QFT(num_qubits=n, do_swaps=False))
    circ = _gates_to_circuit(circuit_gates, n)
    try:
        overlap = abs(complex(ideal_sv.inner(Statevector(circ)))) ** 2
        return float(np.clip(overlap, 0.0, 1.0))
    except Exception:
        return 0.0


def grade_vqe_h2(
    circuit_gates: List[Dict[str, Any]],
    qubit_count: int,
    problem_params: Dict[str, Any],
) -> float:
    """Score based on the H₂ ground‑state energy.

    Only gates acting on qubits ``0`` and ``1`` are considered; they form a
    2‑qubit sub‑circuit whose energy expectation value with the H₂ Hamiltonian
    is computed.  The energy is mapped to a score where ``1.0`` corresponds to the
    exact ground‑state energy.
    """
    if not circuit_gates:
        return 0.0
    circ = QuantumCircuit(2)
    for gate in circuit_gates:
        name = gate.get("name", "")
        qubits = [q for q in gate.get("qubits", []) if q < 2]
        if name in _GATE_MAP and qubits:
            try:
                _GATE_MAP[name](circ, qubits)
            except Exception:
                pass
    try:
        sv = Statevector(circ)
        energy = float(sv.expectation_value(_H2_HAMILTONIAN).real)
        score = 1.0 - abs(energy - _H2_GROUND_ENERGY) / (abs(_H2_GROUND_ENERGY) + 1e-8)
        return float(np.clip(score, 0.0, 1.0))
    except Exception:
        return 0.0


def get_grader(task: str):
    """Return the appropriate grader function for *task*.

    Unknown tasks receive a dummy grader that always returns ``0.0``.
    """
    return {
        "parity-optimization": grade_parity_optimization,
        "shors-factoring": grade_shors_factoring,
        "vqe-h2": grade_vqe_h2,
    }.get(task, lambda *_: 0.0)
