import inference
from client import QuantumEnv
from server.environment import QuantumEnvironment


def test_inference_build_environment_prefers_local_runtime():
    env = inference._build_environment("parity-optimization")
    try:
        assert env.__class__.__name__ == "QuantumOptimizationEnv"
        assert env.task == "parity-optimization"
    finally:
        env.close()


def test_server_environment_reset_can_switch_tasks():
    env = QuantumEnvironment(task="parity-optimization")
    obs = env.reset(task="vqe-h2")
    assert env.env.task == "vqe-h2"
    assert obs.problem_params["molecule"] == "H2"


def test_client_parse_result_preserves_metadata():
    client = QuantumEnv(base_url="http://example.com")
    result = client._parse_result(
        {
            "observation": {
                "circuit_gates": [],
                "qubit_count": 3,
                "current_fidelity": [0.5],
                "problem_params": {},
                "steps_remaining": 10,
            },
            "reward": 1.0,
            "done": True,
            "metadata": {"final_score": 0.91},
        }
    )
    assert getattr(result, "metadata", {}) == {"final_score": 0.91}
