#!/usr/bin/env python3
"""Deterministically evaluate one RL-Games Shadow Hand checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

import hinged_cube_tasks  # noqa: F401  registers Isaac-Repose-HingedCube-Shadow-Direct-v0


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Repose-Cube-Shadow-Direct-v0")
parser.add_argument("--agent", default="rl_games_cfg_entry_point")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--episodes", type=int, default=1024)
parser.add_argument("--seed", type=int, default=20261006)
parser.add_argument("--output", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


def _config_value(agent_cfg: dict, key: str, default):
    params = agent_cfg.get("params", {})
    for section_name in ("config", "env"):
        section = params.get(section_name, {})
        if key in section:
            return section[key]
    return default


def _as_tensor(value, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    return torch.as_tensor(value, device=device)


def _extract_policy_obs(obs):
    """Return the batched policy tensor expected by RL-Games."""
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


def _find_consecutive_successes(value, path: str = "infos"):
    """Find the task metric even when Isaac Lab nests it under ``log``."""
    if not isinstance(value, dict):
        return None
    for key, item in value.items():
        item_path = f"{path}.{key}"
        normalized_key = str(key).lower().replace("-", "_").replace("/", "_")
        if "consecutive_success" in normalized_key:
            return item, item_path
        found = _find_consecutive_successes(item, item_path)
        if found is not None:
            return found
    return None


def _probe_object(unwrapped) -> dict | None:
    """Measure the in-hand object of env 0: local bounds, body masses, inertias and contact material."""
    try:
        import omni.usd
        from pxr import Usd, UsdGeom

        prim = omni.usd.get_context().get_stage().GetPrimAtPath("/World/envs/env_0/object")
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
        )
        bounds = cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
        view = unwrapped.object.root_physx_view
        materials = view.get_material_properties()[0]
        return {
            "bbox_min": [float(v) for v in bounds.GetMin()],
            "bbox_max": [float(v) for v in bounds.GetMax()],
            "masses": view.get_masses()[0].reshape(-1).tolist(),
            "inertias": view.get_inertias()[0].reshape(-1).tolist(),
            "coms": view.get_coms()[0].reshape(-1).tolist(),
            "material": materials.reshape(-1, 3)[0].tolist(),
            "materials_all_shapes": materials.reshape(-1, 3).tolist(),
        }
    except Exception as error:  # the probe is extra information; never fail an evaluation because of it
        print(f"object_probe_error={type(error).__name__}: {error}")
        return None


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: dict) -> None:
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if args_cli.num_envs <= 0 or args_cli.episodes <= 0:
        raise ValueError("--num_envs and --episodes must be positive")

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    config = agent_cfg["params"]["config"]
    config["num_actors"] = args_cli.num_envs
    config["seed"] = args_cli.seed
    config["device"] = args_cli.device
    config["device_name"] = args_cli.device
    agent_cfg["params"]["seed"] = args_cli.seed

    env = None
    try:
        base_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        rl_device = str(_config_value(agent_cfg, "device", args_cli.device))
        clip_obs = float(_config_value(agent_cfg, "clip_observations", math.inf))
        clip_actions = float(_config_value(agent_cfg, "clip_actions", math.inf))
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
        # RL-Games defaults to a single unbatched observation during playback.
        # Isaac Lab already returns [num_envs, obs_dim], so prevent get_action()
        # from adding another leading dimension and flattening the whole batch.
        player.has_batch_dimension = True
        print(f"player_has_batch_dimension={player.has_batch_dimension}")

        object_probe = _probe_object(base_env.unwrapped)
        print(f"object_probe={json.dumps(object_probe)}")

        obs = env.reset()
        device = torch.device(rl_device)
        running_return = torch.zeros(args_cli.num_envs, device=device)
        running_length = torch.zeros(args_cli.num_envs, dtype=torch.int64, device=device)
        completed_returns: list[float] = []
        completed_lengths: list[int] = []
        consecutive_success_samples: list[float] = []
        consecutive_success_source = None
        vector_steps = 0

        with torch.inference_mode():
            while len(completed_returns) < args_cli.episodes and simulation_app.is_running():
                policy_obs = player.obs_to_torch(_extract_policy_obs(obs))
                if vector_steps == 0:
                    print(f"policy_observation_shape={tuple(policy_obs.shape)}")
                if policy_obs.ndim != 2 or policy_obs.shape[0] != args_cli.num_envs:
                    raise RuntimeError(
                        "Expected batched policy observations with shape "
                        f"[{args_cli.num_envs}, obs_dim], got {tuple(policy_obs.shape)}"
                    )
                actions = player.get_action(policy_obs, is_deterministic=True)
                if vector_steps == 0:
                    print(f"policy_action_shape={tuple(actions.shape)}")
                if actions.ndim != 2 or actions.shape[0] != args_cli.num_envs:
                    raise RuntimeError(
                        "Expected batched actions with shape "
                        f"[{args_cli.num_envs}, action_dim], got {tuple(actions.shape)}"
                    )
                obs, rewards, dones, infos = env.step(actions)

                rewards_t = _as_tensor(rewards, device).reshape(-1)
                dones_t = _as_tensor(dones, device).reshape(-1).bool()
                running_return += rewards_t
                running_length += 1
                vector_steps += 1

                done_indices = torch.nonzero(dones_t, as_tuple=False).flatten()
                if done_indices.numel():
                    completed_returns.extend(running_return[done_indices].cpu().tolist())
                    completed_lengths.extend(running_length[done_indices].cpu().tolist())
                    running_return[done_indices] = 0.0
                    running_length[done_indices] = 0

                found_successes = _find_consecutive_successes(infos)
                if found_successes is None:
                    found_successes = _find_consecutive_successes(
                        getattr(base_env.unwrapped, "extras", None), path="env.extras"
                    )
                if found_successes is not None:
                    raw_successes, metric_path = found_successes
                    value = _as_tensor(raw_successes, device).float().mean()
                    consecutive_success_samples.append(float(value.cpu()))
                    if consecutive_success_source is None:
                        consecutive_success_source = metric_path
                        print(f"consecutive_success_source={consecutive_success_source}")

        completed_returns = completed_returns[: args_cli.episodes]
        completed_lengths = completed_lengths[: args_cli.episodes]
        if len(completed_returns) < args_cli.episodes:
            raise RuntimeError(
                f"Simulator stopped after {len(completed_returns)} of {args_cli.episodes} requested episodes"
            )

        returns_t = torch.tensor(completed_returns, dtype=torch.float64)
        lengths_t = torch.tensor(completed_lengths, dtype=torch.float64)
        success_t = torch.tensor(consecutive_success_samples, dtype=torch.float64)
        result = {
            "record_type": "shadow_expert_evaluation",
            "task": args_cli.task,
            "checkpoint": str(checkpoint),
            "deterministic": True,
            "seed": args_cli.seed,
            "num_envs": args_cli.num_envs,
            "episodes": len(completed_returns),
            "vector_steps": vector_steps,
            "mean_episode_reward": float(returns_t.mean()),
            "std_episode_reward": float(returns_t.std(unbiased=False)),
            "min_episode_reward": float(returns_t.min()),
            "max_episode_reward": float(returns_t.max()),
            "mean_episode_length": float(lengths_t.mean()),
            "min_episode_length": int(lengths_t.min()),
            "max_episode_length": int(lengths_t.max()),
            "consecutive_success_source": consecutive_success_source,
            "mean_consecutive_successes": float(success_t.mean()) if success_t.numel() else None,
            "final_consecutive_successes": float(success_t[-1]) if success_t.numel() else None,
            "object_probe": object_probe,
        }
        if hasattr(base_env.unwrapped, "hinge_summary"):
            result["hinge"] = base_env.unwrapped.hinge_summary()

        output_path = Path(args_cli.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for key, value in result.items():
            print(f"{key}={value}")
        print("checkpoint_evaluation_status=PASS")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
