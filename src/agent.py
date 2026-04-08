"""
PyTorch DQN building blocks for quantum circuit optimization.

Public API
----------
OBS_DIM           : int — size of the encoded observation vector (13).
encode_observation : Dict -> Tensor[OBS_DIM] — converts env obs to feature vector.
DQNNetwork        : nn.Module — maps Tensor[B, OBS_DIM] to Q-values Tensor[B, 20].
ReplayBuffer      : circular replay buffer storing (s, a, r, s', done) transitions.
"""

from collections import deque
from typing import Dict, Any, Tuple
import random

import numpy as np
import torch
import torch.nn as nn

# Gate vocabulary used for histogram encoding.
# Index 9 ("other") catches any gate name not in this list.
_GATE_VOCAB = ["h", "cx", "t", "s", "x", "ry", "rz", "rx", "measure", "other"]

# Observation vector size:
#   10 (gate histogram) + 1 (qubit_count) + 1 (fidelity) + 1 (steps) = 13
OBS_DIM: int = len(_GATE_VOCAB) + 3  # = 13


def encode_observation(obs: Dict[str, Any]) -> torch.Tensor:
    """
    Convert an environment observation dict to a float32 Tensor of shape (OBS_DIM,).

    Feature layout:
        [0–9]  Gate-type histogram, normalized by total gate count.
               Index i = count of _GATE_VOCAB[i] gates / total gates.
               Unknown gate names go into index 9 ("other").
        [10]   qubit_count / 10.0
        [11]   current_fidelity[0]   (already in [0, 1])
        [12]   steps_remaining / 50.0
    """
    gates = obs.get("circuit_gates", [])
    hist = np.zeros(len(_GATE_VOCAB), dtype=np.float32)
    for gate in gates:
        name = gate.get("name", "other")
        idx = _GATE_VOCAB.index(name) if name in _GATE_VOCAB else len(_GATE_VOCAB) - 1
        hist[idx] += 1.0
    total = float(max(1, len(gates)))
    hist /= total

    qubit_norm = float(obs.get("qubit_count", 3)) / 10.0
    fidelity = float(obs["current_fidelity"][0])
    steps_norm = float(obs.get("steps_remaining", 50)) / 50.0

    features = np.concatenate([hist, [qubit_norm, fidelity, steps_norm]])
    return torch.from_numpy(features.astype(np.float32))  # shape (13,), dtype float32


class DQNNetwork(nn.Module):
    """
    Two-hidden-layer MLP: observation features → Q-values for 20 actions.

    Architecture
    ------------
    Linear(input_dim → hidden_dim) → ReLU
    Linear(hidden_dim → hidden_dim) → ReLU
    Linear(hidden_dim → n_actions)
    """

    def __init__(self, input_dim: int = OBS_DIM, hidden_dim: int = 128, n_actions: int = 20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: Tensor of shape (B, input_dim) or (input_dim,)
        Returns:
            Q-values of shape (B, n_actions)
        """
        return self.net(x)


class ReplayBuffer:
    """Circular experience replay buffer.

    Stores (state, action, reward, next_state, done) tuples.
    When capacity is exceeded, the oldest transition is overwritten.
    """

    def __init__(self, capacity: int = 10_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor, done: bool) -> None:
        """Add one transition to the buffer."""
        self.buffer.append((state, int(action), float(reward), next_state, bool(done)))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a random batch of transitions.

        Returns
        -------
        states      : Tensor (B, OBS_DIM)
        actions     : Tensor (B,) long
        rewards     : Tensor (B,) float32
        next_states : Tensor (B, OBS_DIM)
        dones       : Tensor (B,) float32  (1.0 = terminal)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)
