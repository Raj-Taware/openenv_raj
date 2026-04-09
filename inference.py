#!/usr/bin/env python3
"""Inference script with mandatory OpenEnv stdout contract.

This keeps the original essence (run all three quantum tasks with HybridPolicy)
while emitting only [START], [STEP], and [END] lines in the required format.
"""

import os
from typing import List

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from src.environment import QuantumOptimizationEnv
from src.policy import HybridPolicy

# Mandatory variables. Defaults are only allowed for API_BASE_URL and MODEL_NAME.
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "quantum-optimization")

# Enable LLM only when API key is configured.
_USE_LLM: bool = bool(HF_TOKEN)

_MODEL_PATHS = {
    "parity-optimization": "models/dqn_parity.pth",
    "shors-factoring": "models/dqn_shors.pth",
    "vqe-h2": "models/dqn_vqe.pth",
}

_ACTION_STRINGS = {
    0: "h(0)",
    1: "h(1)",
    2: "h(2)",
    3: "cx(0,1)",
    4: "cx(0,2)",
    5: "cx(1,0)",
    6: "cx(1,2)",
    7: "cx(2,0)",
    8: "cx(2,1)",
    9: "t(0)",
    10: "t(1)",
    11: "t(2)",
    12: "s(0)",
    13: "s(1)",
    14: "s(2)",
    15: "x(0)",
    16: "x(1)",
    17: "x(2)",
    18: "remove_last_gate()",
    19: "noop()",
}


def _action_to_str(action: int) -> str:
    return _ACTION_STRINGS.get(int(action), f"action({int(action)})")


def _to_bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def _normalize_score(raw_score: float) -> float:
    return min(max(float(raw_score), 0.0), 1.0)


def run_task(task: str) -> float:
    """Run one episode and emit exactly START/STEP/END lines to stdout."""
    # Instantiate OpenAI client using mandatory variables for all LLM calls.
    # The policy receives these same values and uses OpenAI-compatible chat API.
    if _USE_LLM and OpenAI is not None:
        _ = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    env = QuantumOptimizationEnv(task=task)
    policy = HybridPolicy(
        task=task,
        model_path=_MODEL_PATHS.get(task),
        use_llm=_USE_LLM,
        model_name=MODEL_NAME,
        api_base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )

    rewards: List[float] = []
    done = False
    steps_taken = 0
    success = False
    score = 0.0

    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    try:
        obs = env.reset()
        info = {}

        while not done and steps_taken < env.max_steps:
            action, _reasoning = policy.select_action(obs)
            next_obs, reward, done, info = env.step(int(action))

            steps_taken += 1
            reward_value = float(reward)
            rewards.append(reward_value)

            error = info.get("last_action_error") if isinstance(info, dict) else None
            error_text = str(error) if error is not None else "null"

            print(
                f"[STEP] step={steps_taken} action={_action_to_str(int(action))} "
                f"reward={reward_value:.2f} done={_to_bool_text(done)} error={error_text}",
                flush=True,
            )
            obs = next_obs

        raw_score = 0.0
        if isinstance(info, dict) and "final_score" in info:
            raw_score = float(info["final_score"])
        else:
            raw_score = float(env._get_final_score())

        score = _normalize_score(raw_score)
        success = bool(score > 0.0)
        return score

    finally:
        try:
            env.close()
        finally:
            rewards_str = ",".join(f"{r:.2f}" for r in rewards)
            print(
                f"[END] success={_to_bool_text(success)} steps={steps_taken} "
                f"score={score:.2f} rewards={rewards_str}",
                flush=True,
            )


if __name__ == "__main__":
    tasks = ["parity-optimization", "shors-factoring", "vqe-h2"]
    for t in tasks:
        run_task(t)
