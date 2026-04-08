from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import List, Dict, Any

class QuantumAction(Action):
    """Action for the Quantum Optimization environment."""
    action: int = Field(..., description="Action index between 0 and 19")

class QuantumObservation(Observation):
    """Observation from the Quantum Optimization environment."""
    circuit_gates: List[Dict[str, Any]] = Field(default_factory=list, description="Gates built in the circuit")
    qubit_count: int = Field(default=3, description="Number of qubits in circuit")
    current_fidelity: List[float] = Field(default_factory=lambda: [0.0], description="Current overlap with target")
    problem_params: Dict[str, Any] = Field(default_factory=dict, description="Task specific parameters")
    steps_remaining: int = Field(default=50, description="Steps remaining before episode termination")
