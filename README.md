---
title: Quantum Optimization OpenEnv
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---
# Quantum Algorithm Optimization Environment
An OpenEnv environment for optimizing quantum algorithms using reinforcement learning. This environment simulates quantum circuits where an RL agent can modify gates to achieve better performance on real-world tasks like factoring and molecular energy calculations.

## Tasks

- **Easy**: Optimize a 3-qubit parity circuit
- **Medium**: Implement Shor's algorithm for factoring 15
- **Hard**: Use VQE to find H2 ground state energy

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Running the Environment

```python
from src.environment import QuantumOptimizationEnv

env = QuantumOptimizationEnv(task='parity-optimization')
obs = env.reset()
action = 0  # Add H gate
obs, reward, done, info = env.step(action)
```

### Baseline Inference

```bash
python baseline_inference.py
```

This runs a random policy on all tasks and outputs logs in the required format.

## Deployment

Built for Hugging Face Spaces with Docker support.

## Dependencies

- Qiskit: Quantum circuit simulation
- Gym: RL environment interface
- NumPy: Numerical computations

## License

MIT