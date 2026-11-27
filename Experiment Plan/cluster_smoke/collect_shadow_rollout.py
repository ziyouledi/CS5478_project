#!/usr/bin/env python3
"""Collect deterministic Shadow Hand expert transitions into memory-mapped NumPy arrays.

The observation width is the env's (157, or 162 on the unified task). On the unified task the collector also
stores, per row and at the same moment as the observation, the hinge angle, the hinge goal and the phase kind
(hinge_angle.npy, hinge_goal.npy, phase_kind.npy) so the unified intents can be validated offline.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Repose-Cube-Shadow-Direct-v0")
parser.add_argument("--agent", default="rl_games_cfg_entry_point")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--vector_steps", type=int, default=700)
parser.add_argument("--seed", type=int, default=20261007)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from numpy.lib.format import open_memmap
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2_tasks  # noqa: E402,F401  registers Isaac-Repose-Cube-Shadow-ZAxis-Direct-v0
import hinged_cube_tasks  # noqa: E402,F401  registers the face-turn task


ACTION_NAMES = [
    "robot0_WRJ1",
    "robot0_WRJ0",
    "robot0_FFJ3",
    "robot0_MFJ3",
    "robot0_RFJ3",
    "robot0_LFJ4",
    "robot0_THJ4",
    "robot0_FFJ2",
    "robot0_MFJ2",
    "robot0_RFJ2",
    "robot0_LFJ3",
    "robot0_THJ3",
    "robot0_FFJ1",
    "robot0_MFJ1",
    "robot0_RFJ1",
    "robot0_LFJ2",
    "robot0_THJ2",
    "robot0_LFJ1",
    "robot0_THJ1",
    "robot0_THJ0",
]

OBSERVATION_SLICES = {
    "hand_dof_pos": [0, 24],
    "hand_dof_vel": [24, 48],
    "object_pos": [48, 51],
    "object_rot": [51, 55],
    "object_linvel": [55, 58],
    "object_angvel": [58, 61],
    "goal_pos_in_hand": [61, 64],
    "goal_rot": [64, 68],
    "object_to_goal_rot": [68, 72],
    "fingertip_pos": [72, 87],
    "fingertip_rot": [87, 107],
    "fingertip_velocities": [107, 137],
    "previous_action": [137, 157],
}
UNIFIED_OBSERVATION_SLICES = {
    **OBSERVATION_SLICES,
    "sin_cos_hinge_error": [157, 159],
    "is_turn": [159, 160],
    "hold_fraction": [160, 161],
    "hinge_velocity_scaled": [161, 162],
}


def config_value(agent_cfg: dict, key: str, default):
    params = agent_cfg.get("params", {})
    for section_name in ("config", "env"):
        section = params.get(section_name, {})
        if key in section:
            return section[key]
    return default


def extract_policy_obs(obs):
    if isinstance(obs, tuple):
        obs = obs[0]
    if isinstance(obs, dict):
        if "obs" in obs:
            obs = obs["obs"]
        elif "policy" in obs:
            obs = obs["policy"]
        else:
            raise KeyError(f"Cannot find policy observations in keys: {sorted(obs)}")
    return obs


def find_consecutive_successes(value, path: str = "infos"):
    if not isinstance(value, dict):
        return None
    for key, item in value.items():
        item_path = f"{path}.{key}"
        normalized_key = str(key).lower().replace("-", "_").replace("/", "_")
        if "consecutive_success" in normalized_key:
            return item, item_path
        found = find_consecutive_successes(item, item_path)
        if found is not None:
            return found
    return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: dict) -> None:
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if output_dir.exists() or partial_dir.exists():
        raise FileExistsError(f"Output or partial directory already exists: {output_dir}")
    if args_cli.num_envs <= 0 or args_cli.vector_steps <= 0:
        raise ValueError("--num_envs and --vector_steps must be positive")

    total_samples = args_cli.num_envs * args_cli.vector_steps
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    config = agent_cfg["params"]["config"]
    config["num_actors"] = args_cli.num_envs
    config["seed"] = args_cli.seed
    config["device"] = args_cli.device
    config["device_name"] = args_cli.device
    agent_cfg["params"]["seed"] = args_cli.seed

    partial_dir.mkdir(parents=True)
    checkpoint_sha = sha256(checkpoint)
    env = None
    started = time.monotonic()
    try:
        base_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        rl_device = str(config_value(agent_cfg, "device", args_cli.device))
        clip_obs = float(config_value(agent_cfg, "clip_observations", math.inf))
        clip_actions = float(config_value(agent_cfg, "clip_actions", math.inf))
        env = RlGamesVecEnvWrapper(base_env, rl_device, clip_obs, clip_actions)
        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
        )
        env_configurations.register(
            "rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env}
        )

        runner = Runner()
        runner.load(agent_cfg)
        player = runner.create_player()
        player.restore(str(checkpoint))
        player.reset()
        player.has_batch_dimension = True

        obs_dim = int(env.observation_space.shape[0])
        unified = args_cli.task == hinged_cube_tasks.UNIFIED_TASK
        if obs_dim != (162 if unified else 157):
            raise RuntimeError(f"Unexpected observation width {obs_dim} for {args_cli.task}")
        unified_arrays = {}
        if unified:
            unified_arrays = {
                "hinge_angle.npy": open_memmap(partial_dir / "hinge_angle.npy", mode="w+", dtype=np.float32, shape=(total_samples,)),
                "hinge_goal.npy": open_memmap(partial_dir / "hinge_goal.npy", mode="w+", dtype=np.float32, shape=(total_samples,)),
                "phase_kind.npy": open_memmap(partial_dir / "phase_kind.npy", mode="w+", dtype=np.uint8, shape=(total_samples,)),
            }

        observations = open_memmap(
            partial_dir / "observations.npy",
            mode="w+",
            dtype=np.float32,
            shape=(total_samples, obs_dim),
        )
        actions = open_memmap(
            partial_dir / "actions.npy",
            mode="w+",
            dtype=np.float32,
            shape=(total_samples, 20),
        )
        rewards = open_memmap(
            partial_dir / "rewards.npy", mode="w+", dtype=np.float32, shape=(total_samples,)
        )
        dones = open_memmap(
            partial_dir / "dones.npy", mode="w+", dtype=np.bool_, shape=(total_samples,)
        )
        env_ids = open_memmap(
            partial_dir / "env_ids.npy", mode="w+", dtype=np.uint16, shape=(total_samples,)
        )
        episode_ids = open_memmap(
            partial_dir / "episode_ids.npy", mode="w+", dtype=np.uint32, shape=(total_samples,)
        )
        episode_steps = open_memmap(
            partial_dir / "episode_steps.npy", mode="w+", dtype=np.uint16, shape=(total_samples,)
        )

        obs = env.reset()
        per_env_episode = np.zeros(args_cli.num_envs, dtype=np.uint32)
        per_env_step = np.zeros(args_cli.num_envs, dtype=np.uint16)
        per_env_return = np.zeros(args_cli.num_envs, dtype=np.float64)
        per_env_length = np.zeros(args_cli.num_envs, dtype=np.uint32)
        completed_returns: list[float] = []
        completed_lengths: list[int] = []
        success_samples: list[float] = []
        success_source = None
        action_sum = np.zeros(20, dtype=np.float64)
        action_sq_sum = np.zeros(20, dtype=np.float64)
        reward_sum = 0.0
        reward_sq_sum = 0.0
        completed_steps = 0

        with torch.inference_mode():
            for vector_step in range(args_cli.vector_steps):
                if not simulation_app.is_running():
                    raise RuntimeError(f"Simulator stopped after {vector_step} vector steps")
                policy_obs = player.obs_to_torch(extract_policy_obs(obs))
                if policy_obs.shape != (args_cli.num_envs, obs_dim):
                    raise RuntimeError(f"Unexpected observation shape: {tuple(policy_obs.shape)}")
                policy_actions = player.get_action(policy_obs, is_deterministic=True)
                if policy_actions.shape != (args_cli.num_envs, 20):
                    raise RuntimeError(f"Unexpected action shape: {tuple(policy_actions.shape)}")

                obs_cpu = policy_obs.detach().cpu().numpy().astype(np.float32, copy=False)
                action_cpu = policy_actions.detach().cpu().numpy().astype(np.float32, copy=False)
                start = vector_step * args_cli.num_envs
                end = start + args_cli.num_envs
                observations[start:end] = obs_cpu
                actions[start:end] = action_cpu
                env_ids[start:end] = np.arange(args_cli.num_envs, dtype=np.uint16)
                episode_ids[start:end] = per_env_episode
                episode_steps[start:end] = per_env_step
                if unified:
                    unwrapped = base_env.unwrapped
                    unified_arrays["hinge_angle.npy"][start:end] = unwrapped.object.data.joint_pos[:, 0].cpu().numpy()
                    unified_arrays["hinge_goal.npy"][start:end] = unwrapped.hinge_goal.cpu().numpy()
                    unified_arrays["phase_kind.npy"][start:end] = unwrapped.kind.cpu().numpy().astype(np.uint8)

                obs, step_rewards, step_dones, infos = env.step(policy_actions)
                rewards_cpu = step_rewards.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
                dones_cpu = step_dones.detach().cpu().numpy().reshape(-1).astype(np.bool_, copy=False)
                rewards[start:end] = rewards_cpu
                dones[start:end] = dones_cpu

                action_sum += action_cpu.sum(axis=0, dtype=np.float64)
                action_sq_sum += np.square(action_cpu, dtype=np.float64).sum(axis=0)
                reward_sum += rewards_cpu.sum(dtype=np.float64)
                reward_sq_sum += np.square(rewards_cpu, dtype=np.float64).sum()
                per_env_return += rewards_cpu
                per_env_length += 1
                completed_steps += 1

                done_indices = np.flatnonzero(dones_cpu)
                if done_indices.size:
                    completed_returns.extend(per_env_return[done_indices].tolist())
                    completed_lengths.extend(per_env_length[done_indices].tolist())
                    per_env_return[done_indices] = 0.0
                    per_env_length[done_indices] = 0

                found_successes = find_consecutive_successes(infos)
                if found_successes is None:
                    found_successes = find_consecutive_successes(
                        getattr(base_env.unwrapped, "extras", None), path="env.extras"
                    )
                if found_successes is not None:
                    raw_value, metric_path = found_successes
                    value = torch.as_tensor(raw_value, device=rl_device).float().mean()
                    success_samples.append(float(value.cpu()))
                    if success_source is None:
                        success_source = metric_path

                per_env_step += 1
                per_env_episode[dones_cpu] += 1
                per_env_step[dones_cpu] = 0

                if vector_step == 0 or (vector_step + 1) % 100 == 0:
                    print(
                        f"rollout_progress={vector_step + 1}/{args_cli.vector_steps} "
                        f"samples_written={end} completed_episodes={len(completed_returns)}"
                    )

        for array in (observations, actions, rewards, dones, env_ids, episode_ids, episode_steps, *unified_arrays.values()):
            array.flush()

        elapsed = time.monotonic() - started
        action_mean = action_sum / total_samples
        action_variance = np.maximum(action_sq_sum / total_samples - np.square(action_mean), 0.0)
        reward_mean = reward_sum / total_samples
        reward_variance = max(reward_sq_sum / total_samples - reward_mean**2, 0.0)
        manifest = {
            "schema_version": 1,
            "record_type": "shadow_expert_rollout",
            "created_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
            "task": args_cli.task,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "deterministic": True,
            "seed": args_cli.seed,
            "num_envs": args_cli.num_envs,
            "vector_steps": args_cli.vector_steps,
            "total_samples": total_samples,
            "elapsed_seconds": elapsed,
            "samples_per_second": total_samples / elapsed,
            "completed_episodes": len(completed_returns),
            "mean_completed_episode_reward": (
                float(np.mean(completed_returns)) if completed_returns else None
            ),
            "mean_completed_episode_length": (
                float(np.mean(completed_lengths)) if completed_lengths else None
            ),
            "mean_transition_reward": float(reward_mean),
            "std_transition_reward": float(math.sqrt(reward_variance)),
            "done_count": int(np.count_nonzero(dones)),
            "consecutive_success_source": success_source,
            "mean_consecutive_successes": (
                float(np.mean(success_samples)) if success_samples else None
            ),
            "action_names": ACTION_NAMES,
            "action_mean": action_mean.tolist(),
            "action_std": np.sqrt(action_variance).tolist(),
            "observation_slices": UNIFIED_OBSERVATION_SLICES if unified else OBSERVATION_SLICES,
            "unified_env": {key: os.environ.get(key) for key in sorted(os.environ) if key.startswith(("UNIFIED_", "HINGE_", "FACE_TURN_"))} if unified else None,
            "arrays": {
                "observations.npy": {"shape": [total_samples, obs_dim], "dtype": "float32"},
                "actions.npy": {"shape": [total_samples, 20], "dtype": "float32"},
                "rewards.npy": {"shape": [total_samples], "dtype": "float32"},
                "dones.npy": {"shape": [total_samples], "dtype": "bool"},
                "env_ids.npy": {"shape": [total_samples], "dtype": "uint16"},
                "episode_ids.npy": {"shape": [total_samples], "dtype": "uint32"},
                "episode_steps.npy": {"shape": [total_samples], "dtype": "uint16"},
                **{name: {"shape": [total_samples], "dtype": str(array.dtype)} for name, array in unified_arrays.items()},
            },
        }
        write_manifest(partial_dir / "manifest.json", manifest)

        checksums = []
        for file_path in sorted(partial_dir.iterdir()):
            if file_path.name != "SHA256SUMS":
                checksums.append(f"{sha256(file_path)}  {file_path.name}")
        (partial_dir / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
        os.replace(partial_dir, output_dir)

        print(f"rollout_output_dir={output_dir}")
        print(f"rollout_samples={total_samples}")
        print(f"rollout_completed_episodes={len(completed_returns)}")
        print(f"rollout_elapsed_seconds={elapsed:.3f}")
        print(f"rollout_samples_per_second={total_samples / elapsed:.3f}")
        print(f"rollout_mean_episode_reward={manifest['mean_completed_episode_reward']}")
        print(f"rollout_mean_consecutive_successes={manifest['mean_consecutive_successes']}")
        print("rollout_collection_status=PASS")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
