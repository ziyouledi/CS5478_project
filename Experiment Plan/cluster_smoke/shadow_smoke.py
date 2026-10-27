"""Run a finite Shadow Hand random-action smoke test inside Isaac Lab."""

import argparse
import time

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Finite Shadow Hand smoke test")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401  Registers Isaac Lab task IDs.
from isaaclab_tasks.utils import parse_env_cfg


def describe(value):
    if isinstance(value, dict):
        return {key: describe(item) for key, item in value.items()}
    return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}


env = None
try:
    print(f"torch_cuda_available={torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot see the allocated CUDA device")
    print(f"torch_cuda_device={torch.cuda.get_device_name(0)}", flush=True)

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env = gym.make(args.task, cfg=env_cfg)
    obs, _ = env.reset()

    print(f"task={args.task}", flush=True)
    print(f"num_envs={env.unwrapped.num_envs}", flush=True)
    print(f"action_space={env.action_space}", flush=True)
    print(f"observation={describe(obs)}", flush=True)

    start = time.perf_counter()
    with torch.inference_mode():
        for step in range(1, args.steps + 1):
            actions = 2.0 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1.0
            obs, rewards, terminated, truncated, extras = env.step(actions)
            if step == 1:
                print(f"reward_shape={list(rewards.shape)}", flush=True)
                print(f"terminated_shape={list(terminated.shape)}", flush=True)
                print(f"truncated_shape={list(truncated.shape)}", flush=True)
            if step == 1 or step % 50 == 0 or step == args.steps:
                print(f"completed_step={step}", flush=True)
    elapsed = time.perf_counter() - start

    print(f"measured_steps={args.steps}", flush=True)
    print(f"step_elapsed_seconds={elapsed:.6f}", flush=True)
    print(f"environment_steps_per_second={args.steps / elapsed:.3f}", flush=True)
    print(f"aggregate_transitions_per_second={args.steps * env.unwrapped.num_envs / elapsed:.3f}", flush=True)
    print("python_smoke_status=PASS", flush=True)
finally:
    if env is not None:
        env.close()
    simulation_app.close()
