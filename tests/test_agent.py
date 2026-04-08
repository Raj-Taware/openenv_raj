import torch
import numpy as np
import pytest
import os
from src.agent import DQNNetwork, encode_observation, ReplayBuffer, OBS_DIM
from src.train import train_dqn

# A representative observation dict used across multiple tests
SAMPLE_OBS = {
    "circuit_gates": [
        {"name": "h",  "qubits": [0]},
        {"name": "cx", "qubits": [0, 1]},
        {"name": "t",  "qubits": [2]},
    ],
    "qubit_count": 3,
    "current_fidelity": np.array([0.42], dtype=np.float32),
    "problem_params": {"parity_bits": [0, 1, 2]},
    "steps_remaining": 40,
}

# ── encode_observation ────────────────────────────────────────────────────────

def test_encode_observation_returns_tensor():
    out = encode_observation(SAMPLE_OBS)
    assert isinstance(out, torch.Tensor), f"Expected Tensor, got {type(out)}"

def test_encode_observation_shape():
    out = encode_observation(SAMPLE_OBS)
    assert out.shape == (OBS_DIM,), f"Expected ({OBS_DIM},), got {out.shape}"

def test_encode_observation_dtype():
    out = encode_observation(SAMPLE_OBS)
    assert out.dtype == torch.float32

def test_encode_observation_empty_circuit():
    obs = dict(SAMPLE_OBS, circuit_gates=[], current_fidelity=np.array([0.0], dtype=np.float32))
    out = encode_observation(obs)
    assert out.shape == (OBS_DIM,)
    # Gate histogram entries should all be 0
    assert out[:10].sum().item() == 0.0

def test_encode_observation_known_values():
    """Gate histogram is normalized; fidelity and step fraction should be exact."""
    obs = {
        "circuit_gates": [{"name": "h", "qubits": [0]}],  # 1 H gate
        "qubit_count": 3,
        "current_fidelity": np.array([0.8], dtype=np.float32),
        "problem_params": {},
        "steps_remaining": 25,
    }
    out = encode_observation(obs)
    assert abs(out[0].item() - 1.0) < 1e-5, "H gate count normalized to 1.0"
    assert abs(out[11].item() - 0.8) < 1e-5, "Fidelity feature should be 0.8"
    assert abs(out[12].item() - 0.5) < 1e-5, "steps_remaining 25/50 = 0.5"

# ── DQNNetwork ────────────────────────────────────────────────────────────────

def test_dqn_single_forward():
    net = DQNNetwork(input_dim=OBS_DIM, hidden_dim=64, n_actions=20)
    x = torch.randn(1, OBS_DIM)
    out = net(x)
    assert out.shape == (1, 20)

def test_dqn_batch_forward():
    net = DQNNetwork(input_dim=OBS_DIM, hidden_dim=64, n_actions=20)
    x = torch.randn(32, OBS_DIM)
    out = net(x)
    assert out.shape == (32, 20)

def test_dqn_no_nan():
    net = DQNNetwork()
    x = torch.randn(8, OBS_DIM)
    out = net(x)
    assert not torch.isnan(out).any()

# ── ReplayBuffer ───────────────────────────────────────────────────────────────

def test_replay_buffer_push_and_len():
    buf = ReplayBuffer(capacity=50)
    s = torch.randn(OBS_DIM)
    buf.push(s, 3, 1.0, s, False)
    assert len(buf) == 1

def test_replay_buffer_sample_shapes():
    buf = ReplayBuffer(capacity=100)
    s = torch.randn(OBS_DIM)
    for i in range(10):
        buf.push(s, i % 20, float(i) * 0.1, s, i % 5 == 0)
    states, actions, rewards, next_states, dones = buf.sample(5)
    assert states.shape == (5, OBS_DIM)
    assert actions.shape == (5,)
    assert rewards.shape == (5,)
    assert dones.shape == (5,)

def test_replay_buffer_capacity_overflow():
    buf = ReplayBuffer(capacity=5)
    s = torch.randn(OBS_DIM)
    for i in range(10):
        buf.push(s, 0, 0.0, s, False)
    assert len(buf) == 5, "Buffer must not exceed capacity"

# ── Training tests ─────────────────────────────────────────────────────────────

def test_train_dqn_returns_network():
    """train_dqn must return a DQNNetwork in eval mode."""
    net = train_dqn(task="parity-optimization", n_episodes=3, save_path=None, verbose=False)
    assert isinstance(net, DQNNetwork)
    assert not net.training, "Returned network should be in eval mode"

def test_train_dqn_saves_weights():
    """train_dqn with save_path must create a loadable .pth file."""
    path = "models/test_train_parity.pth"
    os.makedirs("models", exist_ok=True)
    train_dqn(task="parity-optimization", n_episodes=3, save_path=path, verbose=False)
    assert os.path.exists(path), f"Expected {path} to be created"
    state = torch.load(path, map_location="cpu")
    assert "net.0.weight" in state, "State dict missing first layer weight"
    os.remove(path)

def test_train_dqn_valid_q_values():
    """After training, DQN should produce finite Q-values for any observation."""
    net = train_dqn(task="shors-factoring", n_episodes=5, save_path=None, verbose=False)
    obs = {
        "circuit_gates": [],
        "qubit_count": 3,
        "current_fidelity": np.array([0.0], dtype=np.float32),
        "problem_params": {},
        "steps_remaining": 50,
    }
    state = encode_observation(obs).unsqueeze(0)
    with torch.no_grad():
        q = net(state)
    assert q.shape == (1, 20)
    assert torch.isfinite(q).all(), "Q-values must be finite after training"
