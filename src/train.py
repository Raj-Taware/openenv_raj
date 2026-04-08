import os
import argparse
import random
from typing import Optional

import torch
import torch.nn.functional as F

from src.environment import QuantumOptimizationEnv
from src.agent import DQNNetwork, ReplayBuffer, encode_observation, OBS_DIM

# Hyperparameters (simple defaults)
N_ACTIONS = 20
BATCH_SIZE = 32
GAMMA = 0.99
LR = 1e-3
BUFFER_CAPACITY = 10000
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 0.995  # decay per episode
TARGET_UPDATE = 5  # episodes

def train_dqn(
    task: str = "parity-optimization",
    n_episodes: int = 200,
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> DQNNetwork:
    """Train a DQN for the given quantum optimization task.

    Parameters
    ----------
    task: str
        One of "parity-optimization", "shors-factoring", "vqe-h2".
    n_episodes: int
        Number of training episodes.
    save_path: Optional[str]
        If provided, the trained network weights are saved to this path.
    verbose: bool
        If True, prints simple progress information.

    Returns
    -------
    DQNNetwork
        The trained network in evaluation mode.
    """
    env = QuantumOptimizationEnv(task=task)
    net = DQNNetwork(input_dim=OBS_DIM, hidden_dim=128, n_actions=N_ACTIONS)
    target_net = DQNNetwork(input_dim=OBS_DIM, hidden_dim=128, n_actions=N_ACTIONS)
    target_net.load_state_dict(net.state_dict())
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(capacity=BUFFER_CAPACITY)

    epsilon = EPS_START
    for episode in range(1, n_episodes + 1):
        obs = env.reset()
        state = encode_observation(obs)
        done = False
        while not done:
            # epsilon‑greedy action selection
            if random.random() < epsilon:
                action = random.randrange(N_ACTIONS)
            else:
                with torch.no_grad():
                    q_values = net(state.unsqueeze(0))
                    action = q_values.argmax(dim=1).item()
            # Step environment
            next_obs, reward, done, info = env.step(action)
            next_state = encode_observation(next_obs)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            # Learn if enough samples
            if len(replay_buffer) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
                # Compute Q(s,a)
                q_pred = net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                # Compute target
                with torch.no_grad():
                    q_next = target_net(next_states).max(1)[0]
                    q_target = rewards + GAMMA * q_next * (1 - dones)
                loss = F.mse_loss(q_pred, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # End of episode
        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        # Update target network periodically
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(net.state_dict())
        if verbose and episode % max(1, n_episodes // 10) == 0:
            print(f"Episode {episode}/{n_episodes}, epsilon={epsilon:.3f}")
    # Evaluation mode
    net.eval()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(net.state_dict(), save_path)
    return net

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN for quantum optimization tasks")
    parser.add_argument("--task", type=str, default="parity-optimization",
                        help="Task name (parity-optimization, shors-factoring, vqe-h2)")
    parser.add_argument("--episodes", type=int, default=200,
                        help="Number of training episodes")
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save model weights (.pth)")
    parser.add_argument("--verbose", action="store_true", help="Print training progress")
    args = parser.parse_args()
    train_dqn(task=args.task, n_episodes=args.episodes, save_path=args.save, verbose=args.verbose)
