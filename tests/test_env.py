import pytest
from src.environment import QuantumOptimizationEnv
from src.graders import grade_parity_optimization, grade_shors_factoring, grade_vqe_h2


def test_env_reset():
    env = QuantumOptimizationEnv(task="parity-optimization")
    obs = env.reset()
    assert "circuit_gates" in obs
    assert obs["qubit_count"] == 3
    assert 0.0 <= obs["current_fidelity"][0] <= 1.0


def test_env_step():
    env = QuantumOptimizationEnv(task="parity-optimization")
    env.reset()
    obs, reward, done, info = env.step(0)  # Add H gate
    assert isinstance(obs, dict)
    assert isinstance(reward, float)
    assert isinstance(done, bool)


def test_graders():
    # Test parity grader
    gates = [{"name": "h", "qubits": [0]}, {"name": "cx", "qubits": [0, 1]}]
    score = grade_parity_optimization(gates, 3, {"parity_bits": [0, 1, 2]})
    assert 0.0 <= score <= 1.0

    # Test Shor's grader
    gates = [
        {"name": "h", "qubits": [0]},
        {"name": "cx", "qubits": [0, 1]},
        {"name": "measure", "qubits": [0]},
    ]
    score = grade_shors_factoring(gates, 4, {"target_number": 15})
    assert 0.0 <= score <= 1.0

    # Test VQE grader
    gates = [{"name": "ry", "qubits": [0]}, {"name": "cx", "qubits": [0, 1]}]
    score = grade_vqe_h2(gates, 4, {"molecule": "H2"})
    assert 0.0 <= score <= 1.0

def test_grade_parity_ideal_circuit_scores_high():
    """CNOT(0→2) + CNOT(1→2) is the ideal parity circuit; must score ≥ 0.95."""
    gates = [{"name": "cx", "qubits": [0, 2]}, {"name": "cx", "qubits": [1, 2]}]
    score = grade_parity_optimization(gates, 3, {"parity_bits": [0, 1, 2]})
    assert score >= 0.95, f"Ideal parity circuit scored {score}, expected ≥0.95"

def test_grade_empty_circuit_scores_near_zero():
    """Empty circuit should score < 0.1 for all tasks."""
    assert grade_parity_optimization([], 3, {}) < 0.1
    assert grade_shors_factoring([], 3, {}) < 0.1
    assert grade_vqe_h2([], 3, {}) < 0.1

def test_grade_scores_in_range():
    """All graders must return float in [0.0, 1.0]."""
    gates = [{"name": "h", "qubits": [0]}, {"name": "cx", "qubits": [0, 1]}]
    for grader in [grade_parity_optimization, grade_shors_factoring, grade_vqe_h2]:
        score = grader(gates, 3, {})
        assert 0.0 <= score <= 1.0, f"{grader.__name__} returned {score}"
