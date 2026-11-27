#!/usr/bin/env python3
"""Record a demo video of one Shadow Hand policy checkpoint (expert or student).

Loads the run the same way as evaluate_students.py (intent, goal mask and bases from
params/intent.json), wraps the env in gymnasium RecordVideo as Isaac Lab's play.py does, and
films environment ``--env_index`` (default 0) with a close-up camera for ``--video_steps`` control
steps (60 Hz) with the deterministic policy. Prints the success count of every episode of the filmed env
inside the clip so the video can be captioned. The filmed env can be chosen by a fixed rule from a first
pass's trace (select_median_env.py); all envs still run, so the episode is the same as in that pass. Rendering only; no metric here replaces the per-episode evaluation.
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
parser.add_argument("--run_dir", default=None, help="Experiment dir with nn/ and, for students, params/intent.json")
parser.add_argument("--checkpoint", default=None, help="Explicit .pth; default: last periodic checkpoint in run_dir/nn")
parser.add_argument("--iteration", type=int, default=None, help="Periodic checkpoint iteration to use instead of the last")
parser.add_argument("--intent", default=None, help="Override when params/intent.json is absent, e.g. blind for the expert")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--env_index", type=int, default=0, help="Env the camera follows and successes are reported for")
parser.add_argument("--seed", type=int, default=20261028)
parser.add_argument("--video_steps", type=int, default=600, help="Control steps to record (600 = one 10 s episode)")
parser.add_argument("--eye", type=float, nargs=3, default=(0.45, -1.05, 0.95), help="Camera position, filmed-env frame")
parser.add_argument("--lookat", type=float, nargs=3, default=(-0.08, -0.40, 0.60), help="Camera target, filmed-env frame")
parser.add_argument("--intent_delay", type=int, default=None,
                    help="Override the intent delay (control steps) for an evaluation-only delay clip; "
                         "default: intent.json value or 0 (as evaluate_students.py)")
parser.add_argument("--turn_checkpoint", default=None,
                    help="Switch task only: face-turn expert driving envs in a turn phase; --checkpoint is then the "
                         "Stage 1 (reorientation) expert driving every other env (two-expert handoff, as evaluate_switch.py)")
parser.add_argument("--name", default=None, help="Video file prefix; default: run or checkpoint name")
parser.add_argument("--output_dir", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import m2_tasks  # noqa: F401  registers Isaac-Repose-Cube-Shadow-ZAxis-Direct-v0
import hinged_cube_tasks  # noqa: F401  registers the hinged-cube and face-turn tasks (need HINGE_* variables)
from hinged_cube_env import TURN_KIND
from intent_env_wrapper import IntentRlGamesVecEnvWrapper
from intent_modules import build_intent, sha256
from student_eval_utils import list_checkpoints


def load_intent_record(run_dir: Path | None) -> dict:
    record_path = run_dir / "params" / "intent.json" if run_dir is not None else None
    if record_path is not None and record_path.is_file():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if args_cli.intent is not None and args_cli.intent != record["intent"]:
            raise ValueError(f"--intent {args_cli.intent} conflicts with {record_path}: {record['intent']}")
        return record
    if args_cli.intent is None:
        raise FileNotFoundError("params/intent.json is missing; pass --intent explicitly")
    return {"intent": args_cli.intent}


def resolve_checkpoint(run_dir: Path | None) -> Path:
    if args_cli.checkpoint:
        return Path(args_cli.checkpoint).expanduser().resolve()
    if run_dir is None:
        raise ValueError("Pass --run_dir or --checkpoint")
    checkpoints = dict(list_checkpoints(run_dir / "nn"))
    if not checkpoints:
        raise FileNotFoundError(f"No periodic checkpoints in {run_dir / 'nn'}")
    iteration = args_cli.iteration if args_cli.iteration is not None else max(checkpoints)
    if iteration not in checkpoints:
        raise ValueError(f"No checkpoint for iteration {iteration}; have {sorted(checkpoints)}")
    return checkpoints[iteration]


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: dict) -> None:
    run_dir = Path(args_cli.run_dir).expanduser().resolve() if args_cli.run_dir else None
    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = resolve_checkpoint(run_dir)
    record = load_intent_record(run_dir)
    pca_basis = record.get("pca_basis")
    if pca_basis and record.get("pca_basis_sha256") and sha256(Path(pca_basis)) != record["pca_basis_sha256"]:
        raise ValueError(f"PCA basis {pca_basis} changed since training")
    name = args_cli.name or (run_dir.name if run_dir is not None else checkpoint.stem)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.viewer.origin_type = "env"
    if not 0 <= args_cli.env_index < args_cli.num_envs:
        raise ValueError(f"--env_index {args_cli.env_index} outside 0..{args_cli.num_envs - 1}")
    env_index = args_cli.env_index
    env_cfg.viewer.env_index = env_index
    env_cfg.viewer.eye = tuple(args_cli.eye)
    env_cfg.viewer.lookat = tuple(args_cli.lookat)
    config = agent_cfg["params"]["config"]
    config["num_actors"] = args_cli.num_envs
    config["device"] = args_cli.device
    config["device_name"] = args_cli.device
    agent_cfg["params"]["seed"] = args_cli.seed
    rl_device = config["device"]

    # The unified task has a 159-D observation, which the intent wrapper (157-D) rejects; its policy is
    # an expert without intent, so it is played through the plain RL-Games wrapper.
    unified = args_cli.task == hinged_cube_tasks.UNIFIED_TASK
    # Two-expert handoff on the switch task: also played through the plain wrapper, as evaluate_switch.py.
    handoff = args_cli.turn_checkpoint is not None
    if handoff and args_cli.task != hinged_cube_tasks.SWITCH_TASK:
        raise ValueError(f"--turn_checkpoint needs --task {hinged_cube_tasks.SWITCH_TASK}")
    if (unified or handoff) and record["intent"] != "blind":
        raise ValueError("The unified and handoff videos are recorded without intent: pass --intent blind")
    plain = unified or handoff
    intent = build_intent(
        record["intent"],
        expert_checkpoint=Path(record["expert_checkpoint"]) if record.get("expert_checkpoint") else None,
        expert_sha256=record.get("expert_checkpoint_sha256"),
        pca_basis=Path(pca_basis) if pca_basis else None,
        obj_lookahead=record.get("obj_lookahead_rad"),
    ).to(rl_device)

    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    base_env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    base_env = gym.wrappers.RecordVideo(
        base_env,
        video_folder=str(output_dir),
        name_prefix=name,
        step_trigger=lambda step: step == 0,
        video_length=args_cli.video_steps,
        disable_logger=True,
    )
    goal_hidden = bool(record.get("goal_hidden", False))
    # Play with the hold the policy was trained under (as evaluate_students.py does).
    intent_hold = int(record.get("intent_hold", 1))
    trained_delay = int(record.get("intent_delay", 0) or 0)
    intent_delay = trained_delay if args_cli.intent_delay is None else args_cli.intent_delay
    if plain:
        env = RlGamesVecEnvWrapper(base_env, rl_device, clip_obs, clip_actions)
    else:
        env = IntentRlGamesVecEnvWrapper(
            base_env, rl_device, clip_obs, clip_actions, intent, goal_hidden=goal_hidden, intent_hold=intent_hold,
            intent_delay=intent_delay, action_hold=int(record.get("action_hold", 1) or 1),
        )
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # Same capture point as the evaluator: filmed env's success counter just before it is zeroed on reset.
    unwrapped = env.unwrapped
    episodes: list[int] = []
    original_reset_idx = unwrapped._reset_idx

    def reset_idx_with_record(env_ids):
        ids = torch.arange(unwrapped.num_envs, device=unwrapped.device) if env_ids is None else env_ids
        if bool((torch.as_tensor(ids, device=unwrapped.device) == env_index).any()):
            episodes.append(int(unwrapped.successes[env_index].item()))
        original_reset_idx(env_ids)

    def make_player(path: Path):
        runner = Runner()
        runner.load(agent_cfg)
        new_player = runner.create_player()
        new_player.restore(str(path))
        new_player.reset()
        new_player.has_batch_dimension = True
        return new_player

    player = make_player(checkpoint)
    turn_checkpoint = Path(args_cli.turn_checkpoint).expanduser().resolve() if handoff else None
    turn_player = make_player(turn_checkpoint) if handoff else None

    obs = env.reset() if plain else env.reset_with_seed(args_cli.seed)  # plain: seeded by env_cfg.seed
    episodes.clear()  # drop the initial reset
    unwrapped._reset_idx = reset_idx_with_record
    # Per-step trace of every env: cube and goal orientation (w, x, y, z) and the success counter, read after
    # each step (after any reset), for checking what a policy does between goals (e.g. continuous rotation).
    trace_object, trace_goal, trace_successes, trace_episode_step = [], [], [], []
    # Face-turn task: hinge angle and hinge goal (rad); goal_rot then holds the hinge goal, object_rot the bottom part.
    has_hinge = hasattr(unwrapped, "hinge_goal")
    trace_hinge, trace_hinge_goal = [], []
    # Unified task: phase kind (index into hinged_cube_env.PHASE_KINDS) and steps since the phase began.
    # Unified and switch task: phase kind (index into hinged_cube_env.PHASE_KINDS) and steps since the phase began.
    has_phase = hasattr(unwrapped, "current_kind") and hasattr(unwrapped, "phase_steps")
    trace_kind, trace_phase_steps = [], []
    with torch.no_grad():
        for _ in range(args_cli.video_steps):
            policy_obs = obs["obs"] if isinstance(obs, dict) else obs
            actions = player.get_action(player.obs_to_torch(policy_obs), is_deterministic=True)
            if handoff:
                # As evaluate_switch.py: per env, the face-turn expert in a turn phase, the Stage 1 expert otherwise.
                turn_actions = turn_player.get_action(turn_player.obs_to_torch(policy_obs), is_deterministic=True)
                is_turn = (unwrapped.current_kind() == TURN_KIND).to(actions.device)
                actions = torch.where(is_turn.unsqueeze(-1), turn_actions, actions)
            obs, _, _, _ = env.step(actions)
            if has_phase:
                trace_kind.append(unwrapped.current_kind().detach().cpu().numpy().copy())
                trace_phase_steps.append(unwrapped.phase_steps.detach().cpu().numpy().copy())
            trace_object.append(unwrapped.object_rot.detach().cpu().numpy().copy())
            trace_goal.append(unwrapped.goal_rot.detach().cpu().numpy().copy())
            trace_successes.append(unwrapped.successes.detach().cpu().numpy().copy())
            trace_episode_step.append(unwrapped.episode_length_buf.detach().cpu().numpy().copy())
            if has_hinge:
                trace_hinge.append(unwrapped.object.data.joint_pos[:, 0].detach().cpu().numpy().copy())
                trace_hinge_goal.append(unwrapped.hinge_goal.detach().cpu().numpy().copy())
    unfinished = int(unwrapped.successes[env_index].item())
    control_hz = 1.0 / unwrapped.step_dt
    env.close()  # flushes the RecordVideo file

    videos = sorted(str(path) for path in output_dir.glob(f"{name}*.mp4"))
    trace_path = output_dir / f"{name}_trace.npz"
    extra = {"hinge_angle": np.stack(trace_hinge), "hinge_goal": np.stack(trace_hinge_goal)} if has_hinge else {}
    if has_phase:
        extra.update(kind=np.stack(trace_kind), phase_steps=np.stack(trace_phase_steps))
    np.savez_compressed(
        trace_path, object_rot=np.stack(trace_object), goal_rot=np.stack(trace_goal),
        successes=np.stack(trace_successes), episode_step=np.stack(trace_episode_step), control_hz=control_hz,
        **extra,
    )
    report = {
        "record_type": "shadow_policy_video",
        "task": args_cli.task,
        "run_dir": str(run_dir) if run_dir is not None else None,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "turn_checkpoint": str(turn_checkpoint) if handoff else None,
        "turn_checkpoint_sha256": sha256(turn_checkpoint) if handoff else None,
        "intent_record": record,
        "goal_hidden": goal_hidden,
        "intent_hold": intent_hold,
        "intent_delay": intent_delay,
        "intent_delay_trained": trained_delay,
        "seed": args_cli.seed,
        "video_steps": args_cli.video_steps,
        "control_hz": control_hz,
        "num_envs": args_cli.num_envs,
        "env_index": env_index,
        "camera": {"origin": f"env {env_index}", "eye": list(args_cli.eye), "lookat": list(args_cli.lookat)},
        "env_completed_episode_successes": episodes,
        "env_successes_in_unfinished_episode": unfinished,
        "videos": videos,
        "trace": str(trace_path),
        "trace_fields": "object_rot, goal_rot [steps, envs, 4] (w, x, y, z); successes, episode_step [steps, envs]; after each step",
    }
    (output_dir / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"video_env{env_index}_episode_successes={episodes} unfinished={unfinished}")
    print(f"video_files={videos}")
    if not videos:
        raise RuntimeError(f"No video written to {output_dir}")
    print("video_record_status=PASS")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Same reasoning as train_student.py: do not hang in simulation_app.close() after a failure.
        traceback.print_exc()
        print("video_record_status=FAIL", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    simulation_app.close()
