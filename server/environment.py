from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import QuantumAction, QuantumObservation
except ImportError:
    from models import QuantumAction, QuantumObservation

from src.environment import QuantumOptimizationEnv

class QuantumEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task: str = "parity-optimization"):
        self.env = QuantumOptimizationEnv(task=task)
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(self, task: str | None = None) -> QuantumObservation:
        if task is not None and task != self.env.task:
            self.env = QuantumOptimizationEnv(task=task)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        obs = self.env.reset()
        
        return QuantumObservation(
            circuit_gates=obs["circuit_gates"],
            qubit_count=int(obs["qubit_count"]),
            current_fidelity=obs["current_fidelity"].tolist(),
            problem_params=obs["problem_params"],
            steps_remaining=int(obs["steps_remaining"]),
            done=False,
            reward=0.0
        )

    def step(self, action: QuantumAction) -> QuantumObservation:
        obs, reward, done, info = self.env.step(action.action)
        self._state.step_count += 1
        
        return QuantumObservation(
            circuit_gates=obs["circuit_gates"],
            qubit_count=int(obs["qubit_count"]),
            current_fidelity=obs["current_fidelity"].tolist(),
            problem_params=obs["problem_params"],
            steps_remaining=int(obs["steps_remaining"]),
            done=done,
            reward=float(reward),
            metadata=info
        )

    @property
    def state(self) -> State:
        return self._state
