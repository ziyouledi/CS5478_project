#!/usr/bin/env python3
"""Train an RL-Games PPO student on Shadow Hand with an intent vector appended to its observation.

Adapted from Isaac Lab v2.3.1 scripts/reinforcement_learning/rl_games/train.py. The environment,
reward, PPO configuration and network are unchanged; only the policy observation grows from
157-D to 157 + intent_dim. The intent is computed from the clipped 157-D observation of the
student's own state (see intent_modules.py).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Repose-Cube-Shadow-Direct-v0")
parser.add_argument("--agent", default="rl_games_cfg_entry_point")
parser.add_argument("--intent", required=True, help="blind | hand-K | wrist-2 | wrist2-finger-K | obj-1|3|4")
parser.add_argument("--expert_checkpoint", default=None)
parser.add_argument("--expert_sha256", default=None)
parser.add_argument("--pca_basis", default=None)
parser.add_argument("--obj_lookahead", type=float, default=None, help="Subgoal lookahead angle in radians.")
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--max_iterations", type=int, required=True)
parser.add_argument("--train_dir", required=True)
parser.add_argument("--experiment_name", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

from intent_modules import OBS_DIM, IntentModule, build_intent


class IntentRlGamesVecEnvWrapper(RlGamesVecEnvWrapper):
    """RL-Games wrapper that appends an intent vector after the standard observation clipping.

    The base wrapper reads the observation space from the unwrapped env, so a gym
    ObservationWrapper cannot change the network input size; this subclass does it here.
    Intent values are not clipped.
    """

    def __init__(self, env, rl_device: str, clip_obs: float, clip_actions: float, intent: IntentModule):
        self._intent = intent
        self._checked_shape = False
        super().__init__(env, rl_device, clip_obs, clip_actions)

    @property
    def observation_space(self):
        base = super().observation_space
        if base.shape != (OBS_DIM,):
            raise ValueError(f"Expected a {OBS_DIM}-D policy observation, got {base.shape}")
        low = np.concatenate([base.low, np.full(self._intent.dim, -np.inf, dtype=np.float32)])
        high = np.concatenate([base.high, np.full(self._intent.dim, np.inf, dtype=np.float32)])
        # The base wrapper type-checks against legacy gym.spaces.Box, not gymnasium's; keep its class.
        return type(base)(low, high, dtype=np.float32)

    def _process_obs(self, obs_dict):
        processed = super()._process_obs(obs_dict)
        obs = processed["obs"]
        intent = self._intent(obs)
        processed["obs"] = torch.cat([obs, intent], dim=1) if self._intent.dim else obs
        if not self._checked_shape:
            print(f"student_obs_shape={tuple(processed['obs'].shape)} intent_dim={self._intent.dim}")
            if processed["obs"].shape[1] != OBS_DIM + self._intent.dim or not torch.isfinite(intent).all():
                raise RuntimeError("Student observation shape or intent values are invalid")
            self._checked_shape = True
        return processed


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: dict) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg["params"]["config"]["device"] = args_cli.device
        agent_cfg["params"]["config"]["device_name"] = args_cli.device
    agent_cfg["params"]["seed"] = args_cli.seed
    agent_cfg["params"]["config"]["max_epochs"] = args_cli.max_iterations
    env_cfg.seed = args_cli.seed

    config = agent_cfg["params"]["config"]
    rollout_batch = args_cli.num_envs * int(config["horizon_length"])
    if rollout_batch % int(config["minibatch_size"]) != 0:
        raise ValueError(f"num_envs * horizon_length = {rollout_batch} is not divisible by minibatch_size")

    rl_device = config["device"]
    intent = build_intent(
        args_cli.intent,
        expert_checkpoint=Path(args_cli.expert_checkpoint) if args_cli.expert_checkpoint else None,
        expert_sha256=args_cli.expert_sha256,
        pca_basis=Path(args_cli.pca_basis) if args_cli.pca_basis else None,
        obj_lookahead=args_cli.obj_lookahead,
    ).to(rl_device)

    train_dir = Path(args_cli.train_dir).expanduser().resolve()
    config["train_dir"] = str(train_dir)
    config["full_experiment_name"] = args_cli.experiment_name
    log_dir = train_dir / args_cli.experiment_name
    env_cfg.log_dir = str(log_dir)
    dump_yaml(str(log_dir / "params" / "env.yaml"), env_cfg)
    dump_yaml(str(log_dir / "params" / "agent.yaml"), agent_cfg)
    intent_record = {
        **intent.describe(),
        "task": args_cli.task,
        "seed": args_cli.seed,
        "num_envs": args_cli.num_envs,
        "max_iterations": args_cli.max_iterations,
        "transitions_per_iteration": rollout_batch,
        "student_obs_dim": OBS_DIM + intent.dim,
    }
    (log_dir / "params" / "intent.json").write_text(json.dumps(intent_record, indent=2) + "\n", encoding="utf-8")
    for key, value in intent_record.items():
        print(f"student_{key}={value}")

    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = IntentRlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, intent)

    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
    config["num_actors"] = env.unwrapped.num_envs

    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)
    runner.reset()
    runner.run({"train": True, "play": False, "sigma": None})
    env.close()
    print(f"student_log_dir={log_dir}")
    print("student_train_status=PASS")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Hydra turns exceptions into SystemExit, and simulation_app.close() can hang after a
        # failed start, which would hold the GPU until the job timeout. Exit immediately instead.
        traceback.print_exc()
        print("student_train_status=FAIL", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    simulation_app.close()
