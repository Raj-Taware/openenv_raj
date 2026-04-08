#!/usr/bin/env python3
"""
Baseline inference script for Quantum Algorithm Optimization Environment.
Runs a simple random policy and logs in required format.
"""

import os
import sys
import json
from src.environment import QuantumOptimizationEnv

# Environment variables for OpenAI client (if needed)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN = os.getenv("HF_TOKEN")


def log_start(task: str):
    print(f"[START] Task: {task}")


def log_step(step: int, action: int, reward: float, done: bool, score: float = None):
    log = {"step": step, "action": action, "reward": reward, "done": done}
    if score is not None:
        log["final_score"] = score
    print(f"[STEP] {json.dumps(log)}")


def log_end(task: str, final_score: float):
    print(f"[END] Task: {task}, Score: {final_score}")


def run_baseline(task: str, max_episodes: int = 1):
    env = QuantumOptimizationEnv(task=task)
    total_reward = 0.0

    log_start(task)

    for episode in range(max_episodes):
        obs = env.reset()
        done = False
        step_count = 0

        while not done and step_count < env.max_steps:
            # Simple random policy
            action = env.action_space.sample()

            next_obs, reward, done, info = env.step(action)
            total_reward += reward

            score = info.get("final_score", None) if done else None
            log_step(step_count, action, reward, done, score)

            obs = next_obs
            step_count += 1

        final_score = env._get_final_score()
        log_end(task, final_score)

    return final_score


if __name__ == "__main__":
    tasks = ["parity-optimization", "shors-factoring", "vqe-h2"]
    results = {}

    for task in tasks:
        score = run_baseline(task)
        results[task] = score

    print("Baseline Results:", json.dumps(results))
