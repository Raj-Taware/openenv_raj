import os
import numpy as np
import pytest
from src.policy import HybridPolicy, LLM_FIDELITY_THRESHOLD

_OBS = {
    "circuit_gates": [
        {"name": "h",  "qubits": [0]},
        {"name": "cx", "qubits": [0, 1]},
    ],
    "qubit_count": 3,
    "current_fidelity": np.array([0.25], dtype=np.float32),
    "problem_params": {"parity_bits": [0, 1, 2]},
    "steps_remaining": 30,
}


def test_policy_dqn_returns_valid_action():
    policy = HybridPolicy("parity-optimization", model_path=None, use_llm=False)
    action, reasoning = policy.select_action(_OBS)
    assert 0 <= action <= 19, f"Action {action} out of [0, 19]"
    assert isinstance(reasoning, str) and len(reasoning) > 0


def test_policy_reasoning_is_string():
    policy = HybridPolicy("parity-optimization", model_path=None, use_llm=False)
    _, reasoning = policy.select_action(_OBS)
    assert isinstance(reasoning, str)


def test_policy_works_for_all_tasks():
    for task in ["parity-optimization", "shors-factoring", "vqe-h2"]:
        policy = HybridPolicy(task, model_path=None, use_llm=False)
        action, _ = policy.select_action(_OBS)
        assert 0 <= action <= 19, f"Invalid action {action} for {task}"


def test_policy_loads_pretrained_weights():
    path = "models/dqn_parity.pth"
    if not os.path.exists(path):
        pytest.skip("Pretrained weights not found — complete Task 6 first")
    policy = HybridPolicy("parity-optimization", model_path=path, use_llm=False)
    action, _ = policy.select_action(_OBS)
    assert 0 <= action <= 19


def test_policy_llm_failure_falls_back_to_dqn():
    policy = HybridPolicy("parity-optimization", model_path=None, use_llm=True)
    policy.llm_client = None  # Simulate LLM unavailable
    action, reasoning = policy.select_action(_OBS)
    assert 0 <= action <= 19
    # Reasoning should indicate fallback to DQN
    assert any(word in reasoning.lower() for word in ["dqn", "policy", "fallback", "failed"])


def test_policy_high_fidelity_uses_dqn():
    obs_high_fidelity = dict(_OBS, current_fidelity=np.array([LLM_FIDELITY_THRESHOLD + 0.1], dtype=np.float32))
    policy = HybridPolicy("parity-optimization", model_path=None, use_llm=True)
    policy.llm_client = None  # Ensure LLM not called
    action, reasoning = policy.select_action(obs_high_fidelity)
    assert 0 <= action <= 19
    assert "dqn" in reasoning.lower()
