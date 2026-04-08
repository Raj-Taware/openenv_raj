from typing import Dict, Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import QuantumAction, QuantumObservation

class QuantumEnv(EnvClient[QuantumAction, QuantumObservation, State]):
    """Client for the Quantum Optimization Environment."""

    def _step_payload(self, action: QuantumAction) -> Dict:
        return {"action": action.action}

    def _parse_result(self, payload: Dict) -> StepResult[QuantumObservation]:
        obs_data = payload.get("observation", {})
        
        # OpenEnv responses wrap the native env correctly
        observation = QuantumObservation(
            circuit_gates=obs_data.get("circuit_gates", []),
            qubit_count=obs_data.get("qubit_count", 3),
            current_fidelity=obs_data.get("current_fidelity", [0.0]),
            problem_params=obs_data.get("problem_params", {}),
            steps_remaining=obs_data.get("steps_remaining", 0),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
