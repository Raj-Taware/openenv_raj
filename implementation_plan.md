# Task 4: PyTorch DQN Components (src/agent.py)

## Goal
Create the DQN building blocks required for the quantum optimization agent, including:
- `OBS_DIM` constant (13)
- `encode_observation` function to convert env observations into a fixed‑size tensor
- `DQNNetwork` class (2‑layer MLP with ReLU, hidden dim 128 by default)
- `ReplayBuffer` class for experience replay
- Corresponding unit tests in `tests/test_agent.py`

## Proposed Changes

### [NEW] src/agent.py
- Implements the observation encoder, network, and replay buffer as described in the plan.
- Uses only standard PyTorch and Python libraries; no external dependencies.

### [NEW] tests/test_agent.py
- Contains 13 tests covering:
  - Observation encoding (type, shape, dtype, empty circuit, known values)
  - DQNNetwork forward passes (single and batch, no NaNs)
  - ReplayBuffer push, length, sampling shapes, capacity overflow.

## Open Questions
- None; the specifications are fully defined in the plan.

## Verification Plan
- Run `pytest -q tests/test_agent.py` to ensure all 13 tests pass.
- Run the full test suite `pytest -q` to confirm no regressions.
- Commit the new files.
