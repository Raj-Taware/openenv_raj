import gradio as gr
from src.environment import QuantumOptimizationEnv
import json


def run_simulation(task: str, steps: int):
    env = QuantumOptimizationEnv(task=task, max_steps=steps)
    obs = env.reset()
    log = "[START] Task: {}\n".format(task)

    for step in range(steps):
        action = env.action_space.sample()  # Random action for demo
        next_obs, reward, done, info = env.step(action)
        log += "[STEP] {}\n".format(
            json.dumps({"step": step, "action": action, "reward": reward, "done": done})
        )
        if done:
            break
        obs = next_obs

    final_score = env._get_final_score()
    log += "[END] Task: {}, Score: {}\n".format(task, final_score)
    return log


with gr.Blocks() as demo:
    gr.Markdown("# Quantum Algorithm Optimization Environment")
    gr.Markdown("Simulate quantum circuit optimization with RL.")

    with gr.Row():
        task = gr.Dropdown(
            ["parity-optimization", "shors-factoring", "vqe-h2"],
            label="Task",
            value="parity-optimization",
        )
        steps = gr.Slider(1, 50, value=10, label="Max Steps")

    btn = gr.Button("Run Simulation")
    output = gr.Textbox(label="Simulation Log", lines=20)

    btn.click(run_simulation, inputs=[task, steps], outputs=output)

if __name__ == "__main__":
    demo.launch()
