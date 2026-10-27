#!/usr/bin/env python3
"""Evaluate every periodic checkpoint of one Shadow Hand student run with per-episode success counts.

One deterministic episode per environment (the first one), with every checkpoint reset to the same
seed. Success count per episode is read from the environment's ``successes`` counter just before it
is zeroed on reset, so it does not depend on the EMA ``consecutive_successes`` metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Repose-Cube-Shadow-Direct-v0")
parser.add_argument("--agent", default="rl_games_cfg_entry_point")
parser.add_argument("--run_dir", required=True, help="Experiment dir containing nn/ and, for students, params/intent.json")
parser.add_argument("--intent", default=None, help="Override when params/intent.json is absent, e.g. blind for the expert")
parser.add_argument("--checkpoint_stride", type=int, default=1)
parser.add_argument("--num_envs", type=int, default=1024, help="Episodes per checkpoint (one per environment)")
parser.add_argument("--seed", type=int, default=20261028)
parser.add_argument("--output_dir", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from isaaclab_rl.rl_games import RlGamesGpuEnv
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

from intent_env_wrapper import IntentRlGamesVecEnvWrapper
from intent_modules import build_intent, sha256
from student_eval_utils import FirstEpisodeTracker, list_checkpoints


def load_intent_record(run_dir: Path) -> dict:
    record_path = run_dir / "params" / "intent.json"
    if record_path.is_file():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if args_cli.intent is not None and args_cli.intent != record["intent"]:
            raise ValueError(f"--intent {args_cli.intent} conflicts with {record_path}: {record['intent']}")
        return record
    if args_cli.intent is None:
        raise FileNotFoundError(f"{record_path} is missing; pass --intent explicitly")
    return {"intent": args_cli.intent}


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: dict) -> None:
    run_dir = Path(args_cli.run_dir).expanduser().resolve()
    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = list_checkpoints(run_dir / "nn", args_cli.checkpoint_stride)
    if not checkpoints:
        raise FileNotFoundError(f"No periodic checkpoints in {run_dir / 'nn'}")

    record = load_intent_record(run_dir)
    pca_basis = record.get("pca_basis")
    if pca_basis and record.get("pca_basis_sha256") and sha256(Path(pca_basis)) != record["pca_basis_sha256"]:
        raise ValueError(f"PCA basis {pca_basis} changed since training")

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    config = agent_cfg["params"]["config"]
    config["num_actors"] = args_cli.num_envs
    config["device"] = args_cli.device
    config["device_name"] = args_cli.device
    agent_cfg["params"]["seed"] = args_cli.seed
    rl_device = config["device"]

    intent = build_intent(
        record["intent"],
        expert_checkpoint=Path(record["expert_checkpoint"]) if record.get("expert_checkpoint") else None,
        expert_sha256=record.get("expert_checkpoint_sha256"),
        pca_basis=Path(pca_basis) if pca_basis else None,
        obj_lookahead=record.get("obj_lookahead_rad"),
    ).to(rl_device)

    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    base_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = IntentRlGamesVecEnvWrapper(base_env, rl_device, clip_obs, clip_actions, intent)
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    unwrapped = env.unwrapped
    holder: dict[str, FirstEpisodeTracker | None] = {"tracker": None}
    original_reset_idx = unwrapped._reset_idx

    def reset_idx_with_record(env_ids):
        tracker = holder["tracker"]
        if tracker is not None:
            ids = torch.arange(unwrapped.num_envs, device=unwrapped.device) if env_ids is None else env_ids
            tracker.record_successes(torch.as_tensor(ids, device=tracker.successes.device),
                                     unwrapped.successes[ids].to(tracker.successes.device))
        original_reset_idx(env_ids)

    unwrapped._reset_idx = reset_idx_with_record
    max_steps = int(unwrapped.max_episode_length) + 10

    runner = Runner()
    runner.load(agent_cfg)
    player = runner.create_player()
    player.has_batch_dimension = True

    results = []
    for iteration, checkpoint in checkpoints:
        started = time.monotonic()
        player.restore(str(checkpoint))
        player.reset()
        player.has_batch_dimension = True
        holder["tracker"] = None
        obs = env.reset_with_seed(args_cli.seed)
        tracker = FirstEpisodeTracker(args_cli.num_envs, rl_device)
        tracker.active = True
        holder["tracker"] = tracker
        steps = 0
        with torch.inference_mode():
            while not tracker.all_finished:
                if steps >= max_steps or not simulation_app.is_running():
                    raise RuntimeError(f"{int((~tracker.finished).sum())} envs unfinished after {steps} steps")
                policy_obs = player.obs_to_torch(obs["obs"])
                actions = player.get_action(policy_obs, is_deterministic=True)
                obs, rewards, dones, extras = env.step(actions)
                tracker.step(rewards.to(rl_device), dones.to(rl_device), extras.get("time_outs"))
                steps += 1
        holder["tracker"] = None
        summary = tracker.summary()
        entry = {
            "iteration": iteration,
            "checkpoint": str(checkpoint),
            "vector_steps": steps,
            "elapsed_seconds": time.monotonic() - started,
            **summary,
        }
        results.append(entry)
        print(
            f"eval iteration={iteration} successes={entry['mean_successes_per_episode']:.3f} "
            f"reward={entry['mean_episode_reward']:.1f} length={entry['mean_episode_length']:.1f} "
            f"fall={entry['fall_fraction']:.3f} elapsed_s={entry['elapsed_seconds']:.1f}"
        )

    report = {
        "schema_version": 1,
        "record_type": "shadow_student_checkpoint_evaluation",
        "task": args_cli.task,
        "run_dir": str(run_dir),
        "intent_record": record,
        "intent_dim": intent.dim,
        "deterministic": True,
        "seed": args_cli.seed,
        "episodes_per_checkpoint": args_cli.num_envs,
        "episode_rule": "first episode of each environment",
        "success_source": "env.successes captured before _reset_idx",
        "checkpoint_stride": args_cli.checkpoint_stride,
        "checkpoints": results,
    }
    (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "curve.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        columns = ["iteration", "mean_successes_per_episode", "std_successes_per_episode",
                   "fraction_episodes_with_success", "mean_episode_reward", "std_episode_reward",
                   "mean_episode_length", "fall_fraction"]
        writer.writerow(columns)
        for entry in results:
            writer.writerow([entry[column] for column in columns])
    env.close()
    print(f"evaluation_output_dir={output_dir}")
    print(f"evaluated_checkpoints={len(results)}")
    print("student_evaluation_status=PASS")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Same reasoning as train_student.py: do not hang in simulation_app.close() after a failure.
        traceback.print_exc()
        print("student_evaluation_status=FAIL", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    simulation_app.close()
