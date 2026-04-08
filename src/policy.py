import os
import json
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch

from src.agent import DQNNetwork, encode_observation, OBS_DIM

# Optional OpenAI import – if unavailable we simply skip LLM usage
try:
    import openai
except ImportError:  # pragma: no cover
    openai = None

# LLM is only queried when fidelity is below this threshold
LLM_FIDELITY_THRESHOLD: float = 0.40

_TASK_DESCRIPTIONS: Dict[str, str] = {
    "parity-optimization": (
        "computing the parity (XOR) of 3 qubits. "
        "Ideal circuit: CNOT(qubit 0 → qubit 2), then CNOT(qubit 1 → qubit 2). "
        "This accumulates the XOR of all three qubits into qubit 2."
    ),
    "shors-factoring": (
        "implementing a 3-qubit Quantum Fourier Transform (QFT), "
        "which is the core period-finding subroutine of Shor's algorithm for factoring 15. "
        "Use H gates and T/S phase gates to approximate the QFT."
    ),
    "vqe-h2": (
        "finding the H2 molecular ground-state energy (-1.137 Hartree) using VQE. "
        "Build a variational ansatz on qubits 0 and 1 with rotation (H, T, S) and CNOT gates. "
        "The energy expectation is computed using the STO-3G Jordan-Wigner Hamiltonian."
    ),
}

_ACTION_GUIDE = """
Actions (choose one integer 0–19):
  0: H on qubit 0     1: H on qubit 1     2: H on qubit 2
  3: CNOT 0→1         4: CNOT 0→2         5: CNOT 1→0
  6: CNOT 1→2         7: CNOT 2→0         8: CNOT 2→1
  9: T on qubit 0    10: T on qubit 1    11: T on qubit 2
 12: S on qubit 0    13: S on qubit 1    14: S on qubit 2
 15: X on qubit 0    16: X on qubit 1    17: X on qubit 2
 18: Remove last gate
 19: No-op"""


class HybridPolicy:
    """Hybrid policy that combines a DQN with optional LLM reasoning.

    The policy prefers the LLM when the current fidelity is low (below
    ``LLM_FIDELITY_THRESHOLD``) *and* an LLM client is available.  Any failure
    in the LLM path – missing client, network error, malformed response, or an
    out‑of‑range action – silently falls back to the DQN.
    """

    def __init__(self, task: str, model_path: Optional[str] = None, use_llm: bool = True, model_name: str = "gpt-3.5-turbo", api_base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.task = task
        self.use_llm = use_llm
        self.model_name = model_name
        # Initialise DQN – random weights if no checkpoint is supplied
        self.dqn = DQNNetwork()
        if model_path and os.path.exists(model_path):
            try:
                state = torch.load(model_path, map_location="cpu")
                # ``train_dqn`` saves only the state dict under key "net"
                if isinstance(state, dict) and "net" in state:
                    self.dqn.load_state_dict(state["net"])
                else:
                    self.dqn.load_state_dict(state)
            except Exception as exc:  # pragma: no cover – defensive
                print(f"[HybridPolicy] Warning: failed to load weights from {model_path}: {exc}")
        self.dqn.eval()
        # LLM client – may be None if openai is not installed or API key missing
        self.llm_client = None
        if openai is not None and self.use_llm:
            api_key_to_use = api_key or os.getenv("OPENAI_API_KEY", "dummy_key")
            base_url_to_use = api_base_url or "https://api.openai.com/v1"
            
            try:
                # Attempt openai >= 1.0.0 client initialization
                self.llm_client = openai.OpenAI(
                    base_url=base_url_to_use,
                    api_key=api_key_to_use
                )
            except AttributeError:
                # Fallback for openai < 1.0.0
                openai.api_base = base_url_to_use
                openai.api_key = api_key_to_use
                self.llm_client = openai

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def select_action(self, obs: Dict[str, Any]) -> Tuple[int, str]:
        """Return ``(action, reasoning)`` for the given observation.

        ``action`` is an integer in ``[0, 19]``. ``reasoning`` is a free‑form
        string explaining the decision – either the LLM's answer or a short
        fallback note.
        """
        fidelity = float(obs.get("current_fidelity", np.array([0.0]))[0])
        # Decide whether to attempt LLM query
        if self.use_llm and fidelity < LLM_FIDELITY_THRESHOLD and self.llm_client is not None:
            try:
                action, reasoning = self._query_llm(obs)
                if 0 <= action <= 19:
                    return action, reasoning
                else:
                    # Out‑of‑range – fall back
                    fallback_reason = f"LLM suggested out‑of‑range action {action}; falling back to DQN."
            except Exception as exc:  # pragma: no cover – any LLM failure
                fallback_reason = f"LLM query failed ({exc}); falling back to DQN."
        else:
            fallback_reason = "LLM not used (high fidelity or disabled). Falling back to DQN."

        # DQN fallback path
        state_tensor = encode_observation(obs).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.dqn(state_tensor)
            action = int(q_vals.argmax(dim=1).item())
        reasoning = fallback_reason
        return action, reasoning

    # ---------------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------------
    def _query_llm(self, obs: Dict[str, Any]) -> Tuple[int, str]:
        """Query the LLM for an action and reasoning.

        The LLM is prompted with a concise description of the task, the action
        guide, and a JSON‑friendly representation of the observation.  The
        expected response is a JSON object with ``"action"`` (int) and ``"reasoning"``
        (string) fields.
        """
        # Build a minimal observation summary for the prompt
        obs_summary = json.dumps(
            {
                "circuit_gates": obs.get("circuit_gates", []),
                "qubit_count": obs.get("qubit_count", 3),
                "current_fidelity": float(obs.get("current_fidelity", np.array([0.0]))[0]),
                "steps_remaining": obs.get("steps_remaining", 0),
            },
            indent=2,
        )
        system_msg = (
            "You are an expert quantum circuit designer. Given the current "
            "state of a quantum circuit, suggest the next gate to apply."
        )
        user_msg = (
            f"Task: {self.task}\n"
            f"Description: {_TASK_DESCRIPTIONS.get(self.task, '')}\n"
            f"Action guide:{_ACTION_GUIDE}\n"
            f"Observation (JSON):\n{obs_summary}\n"
            "Respond with a JSON object containing two fields:\n"
            "  \"action\": <int 0‑19>,\n"
            "  \"reasoning\": <short explanation>."
        )
        if hasattr(self.llm_client, "chat"):
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                temperature=0.0,
            )
        else:
            response = self.llm_client.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                temperature=0.0,
            )
        content = response.choices[0].message.content.strip()
        # Attempt to parse JSON – be tolerant of surrounding text
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract the JSON substring
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1:
                payload = json.loads(content[start : end + 1])
            else:
                raise ValueError("LLM response is not valid JSON")
        action = int(payload.get("action"))
        reasoning = str(payload.get("reasoning", ""))
        return action, reasoning
