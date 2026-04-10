import os
from typing import Dict, Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import QuantumAction, QuantumObservation
except ImportError:
    from models import QuantumAction, QuantumObservation

class QuantumEnv(EnvClient[QuantumAction, QuantumObservation, State]):
    """Client for the Quantum Optimization Environment."""

    def __init__(self, base_url: str | None = None, **kwargs):
        resolved_base_url = base_url or os.getenv("OPENENV_BASE_URL", "http://127.0.0.1:7860")
        super().__init__(base_url=resolved_base_url, **kwargs)

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

        result = StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )
        # Preserve server metadata so inference can read terminal final_score values.
        setattr(result, "metadata", payload.get("metadata", {}))
        return result

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
