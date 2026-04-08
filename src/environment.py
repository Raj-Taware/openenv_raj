"""
Quantum Algorithm Optimization Environment.

An OpenEnv‑compatible Gymnasium environment where an agent modifies a 3‑qubit
quantum circuit to solve one of three tasks by selecting gate‑addition actions.

Fidelity is computed via Qiskit Statevector (exact, no shots, instant for 3
qubits). The environment provides a cached fidelity value that is only
recomputed when the circuit depth changes.
"""

import gymnasium as gym
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.circuit.library import QFT
from typing import Dict, Any, Tuple, Optional
from gymnasium import spaces
from .graders import get_grader


# ---------------------------------------------------------------------------
# Pre‑compute ideal target statevectors once at module load (< 1 ms each)
# ---------------------------------------------------------------------------

def _build_parity_ideal() -> Statevector:
    """3‑qubit parity circuit: CNOT(0→2) then CNOT(1→2) accumulates XOR into qubit 2."""
    circ = QuantumCircuit(3)
    circ.cx(0, 2)
    circ.cx(1, 2)
    return Statevector(circ)


def _build_shors_ideal() -> Statevector:
    """3‑qubit QFT (no final swap) — the core subroutine of Shor's algorithm."""
    return Statevector(QFT(num_qubits=3, do_swaps=False))


PARITY_IDEAL_SV: Statevector = _build_parity_ideal()
SHORS_IDEAL_SV: Statevector = _build_shors_ideal()

# 2‑qubit H₂ Hamiltonian (STO‑3G basis, Jordan‑Wigner transform).
H2_HAMILTONIAN = SparsePauliOp.from_list([
    ("II", -0.8105),
    ("ZI",  0.1722),
    ("IZ", -0.2228),
    ("ZZ",  0.1208),
    ("YY", -0.0453),
    ("XX",  0.1722),
])
H2_GROUND_ENERGY = -1.137  # Hartree, FCI/STO‑3G


class QuantumOptimizationEnv(gym.Env):
    """OpenEnv for quantum algorithm optimization.

    The agent repeatedly chooses one of 20 discrete gate actions to build/modify
    a 3‑qubit quantum circuit.  Each task has a different fidelity target.
    """

    metadata = {"render_modes": []}

    def __init__(self, task: str = "parity-optimization", max_steps: int = 50):
        super().__init__()
        assert task in ("parity-optimization", "shors-factoring", "vqe-h2"), \
            f"Unknown task '{task}'. Choose from: parity-optimization, shors-factoring, vqe-h2"

        self.task = task
        self.max_steps = max_steps
        self.current_step: int = 0
        self.qubit_count: int = 3  # Uniform across all tasks
        self.circuit: Optional[QuantumCircuit] = None
        self.problem_params: Dict[str, Any] = self._get_problem_params()

        # Fidelity cache – keyed by circuit depth (invalidated on any gate change)
        self._cached_fidelity: Optional[float] = None
        self._cached_depth: int = -1

        # ------------------------------------------------------------------
        # Action space: 20 discrete actions
        #   0–2   : Add H gate to qubit 0, 1, 2
        #   3–8   : Add CNOT (6 ordered qubit‑pair combinations of {0,1,2})
        #   9–11  : Add T gate to qubit 0, 1, 2
        #   12–14 : Add S gate to qubit 0, 1, 2
        #   15–17 : Add X gate to qubit 0, 1, 2
        #   18    : Remove last gate (no‑op if circuit empty)
        #   19    : No‑op
        # ------------------------------------------------------------------
        self.action_space = spaces.Discrete(20)

        self.observation_space = spaces.Dict({
            "circuit_gates": spaces.Sequence(
                spaces.Dict({
                    "name":   spaces.Text(10),
                    "qubits": spaces.Sequence(spaces.Discrete(10)),
                })
            ),
            "qubit_count":      spaces.Discrete(10),
            "current_fidelity": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "problem_params":   spaces.Dict({"value": spaces.Box(low=0.0, high=1.0, shape=(1,))}),
            "steps_remaining":  spaces.Discrete(51),
        })

        self.reset()

    # ------------------------------------------------------------------
    # Core Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        super().reset(seed=seed)
        self.current_step = 0
        self.circuit = QuantumCircuit(self.qubit_count)
        self._invalidate_cache()

        # Seed each task with a minimal starting circuit so the agent begins
        # from a non‑trivial state.
        if self.task == "parity-optimization":
            self.circuit.h(0)
        elif self.task == "shors-factoring":
            self.circuit.h(0)
            self.circuit.h(1)
        elif self.task == "vqe-h2":
            self.circuit.x(0)

        self._invalidate_cache()
        return self._get_observation()

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self.current_step += 1
        info: Dict[str, Any] = {}

        valid = self._apply_action(action)
        penalty = -0.1 if not valid else 0.0

        fidelity = self._compute_fidelity()          # uses cache — called only once here
        reward = penalty + (fidelity - 0.5) * 2.0   # maps [0,1] → [-1, 1]

        terminated = bool(fidelity >= 0.95)
        truncated  = bool(self.current_step >= self.max_steps)
        done       = terminated or truncated

        if done:
            info["final_score"] = self._get_final_score()

        # Legacy 4‑tuple compatibility (some callers unpack 4 values)
        observation = self._get_observation()
        return observation, reward, done, info

    # ------------------------------------------------------------------
    # Fidelity (with caching)
    # ------------------------------------------------------------------

    def _invalidate_cache(self) -> None:
        self._cached_fidelity = None
        self._cached_depth = -1

    def _compute_fidelity_uncached(self) -> float:
        """Compute fidelity via Statevector — exact and instant for 3‑qubit circuits."""
        try:
            current_sv = Statevector(self.circuit)
        except Exception:
            return 0.0

        if self.task == "parity-optimization":
            overlap = abs(complex(PARITY_IDEAL_SV.inner(current_sv))) ** 2
            return float(np.clip(overlap, 0.0, 1.0))

        if self.task == "shors-factoring":
            overlap = abs(complex(SHORS_IDEAL_SV.inner(current_sv))) ** 2
            return float(np.clip(overlap, 0.0, 1.0))

        if self.task == "vqe-h2":
            # Build 2‑qubit subcircuit from gates acting only on qubits 0 and 1
            circ_2q = QuantumCircuit(2)
            for instr in self.circuit.data:
                indices = [self.circuit.find_bit(q).index for q in instr.qubits]
                if all(idx < 2 for idx in indices):
                    new_qubits = [circ_2q.qubits[idx] for idx in indices]
                    circ_2q.append(instr.operation, new_qubits)
            try:
                sv_2q = Statevector(circ_2q)
                energy = float(sv_2q.expectation_value(H2_HAMILTONIAN).real)
                score = 1.0 - abs(energy - H2_GROUND_ENERGY) / (abs(H2_GROUND_ENERGY) + 1e-8)
                return float(np.clip(score, 0.0, 1.0))
            except Exception:
                return 0.0

        return 0.5

    def _compute_fidelity(self) -> float:
        """Return fidelity, reusing cached value when circuit depth hasn't changed."""
        depth = len(self.circuit.data)
        if self._cached_fidelity is not None and self._cached_depth == depth:
            return self._cached_fidelity
        self._cached_fidelity = self._compute_fidelity_uncached()
        self._cached_depth = depth
        return self._cached_fidelity

    # ------------------------------------------------------------------
    # Action application
    # ------------------------------------------------------------------

    def _apply_action(self, action: int) -> bool:
        """Modify circuit according to action. Returns True if valid."""
        try:
            if   action == 0:  self.circuit.h(0)
            elif action == 1:  self.circuit.h(1)
            elif action == 2:  self.circuit.h(2)
            elif action == 3:  self.circuit.cx(0, 1)
            elif action == 4:  self.circuit.cx(0, 2)
            elif action == 5:  self.circuit.cx(1, 0)
            elif action == 6:  self.circuit.cx(1, 2)
            elif action == 7:  self.circuit.cx(2, 0)
            elif action == 8:  self.circuit.cx(2, 1)
            elif action == 9:  self.circuit.t(0)
            elif action == 10: self.circuit.t(1)
            elif action == 11: self.circuit.t(2)
            elif action == 12: self.circuit.s(0)
            elif action == 13: self.circuit.s(1)
            elif action == 14: self.circuit.s(2)
            elif action == 15: self.circuit.x(0)
            elif action == 16: self.circuit.x(1)
            elif action == 17: self.circuit.x(2)
            elif action == 18:
                if len(self.circuit.data) > 0:
                    self.circuit.data.pop()
            # action == 19: no‑op (intentional pass)
            self._invalidate_cache()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Observation and scoring
    # ------------------------------------------------------------------

    def _get_observation(self) -> Dict[str, Any]:
        gates = [
            {
                "name":   instr.operation.name,
                "qubits": [self.circuit.find_bit(q).index for q in instr.qubits],
            }
            for instr in self.circuit.data
        ]
        return {
            "circuit_gates":    gates,
            "qubit_count":      self.qubit_count,
            "current_fidelity": np.array([self._compute_fidelity()], dtype=np.float32),
            "problem_params":   self.problem_params,
            "steps_remaining":  self.max_steps - self.current_step,
        }

    def _get_final_score(self) -> float:
        grader = get_grader(self.task)
        obs = self._get_observation()
        return grader(obs["circuit_gates"], obs["qubit_count"], self.problem_params)

    def _get_problem_params(self) -> Dict[str, Any]:
        if self.task == "parity-optimization":
            return {"parity_bits": [0, 1, 2]}
        if self.task == "shors-factoring":
            return {"target_number": 15}
        if self.task == "vqe-h2":
            return {"molecule": "H2", "ground_energy": H2_GROUND_ENERGY}
        return {}

    def state(self) -> Dict[str, Any]:
        """Alias for _get_observation() — OpenEnv spec compatibility."""
        return self._get_observation()
