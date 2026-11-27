#!/usr/bin/env python3
"""Record clean single-hand (num_envs=1) high-camera Task 3 clips using the 6-colour Rubik's cube
(ShadowHandUnifiedZ1Env) so Task 3 matches Task 1/2 framing and Act 4/Teaser cube visuals.

Supports recording either:
  - Unified v4 expert (`--unified_checkpoint`)
  - E3 unified goal-hidden student (`--student_run_dir`, wrapped with IntentRlGamesVecEnvWrapper unified=True)

Saves:
  - `<name>_high.mp4` (60 fps, 600x600 single-hand high-camera view matching Task 1/2 angle)
  - `<name>_trace.json` (per-step completed phases/turns, falls, metadata)
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
parser.add_argument("--task", default="Isaac-Unified-HingedCube-Z1-Shadow-Direct-v0")
parser.add_argument("--agent", default="rl_games_cfg_entry_point")
parser.add_argument("--unified_checkpoint", default=None)
parser.add_argument("--student_run_dir", default=None)
parser.add_argument("--name", required=True)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--seed", type=int, default=20261028)
parser.add_argument("--antialiasing", default="FXAA")
parser.add_argument("--output_dir", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0], *hydra_args]

if bool(args_cli.unified_checkpoint) == bool(args_cli.student_run_dir):
    raise SystemExit("Pass exactly one of --unified_checkpoint or --student_run_dir")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import imageio
import numpy as np
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import hinged_cube_tasks  # noqa: F401
import z1_tasks  # noqa: F401
from hinged_cube_env import PHASE_KINDS, TURN_KIND
from intent_env_wrapper import IntentRlGamesVecEnvWrapper
from intent_modules import build_intent
from student_eval_utils import list_checkpoints


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: dict) -> None:
    out_dir = Path(args_cli.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    student_record = None
    if args_cli.student_run_dir:
        run_dir = Path(args_cli.student_run_dir).expanduser().resolve()
        student_record = json.loads((run_dir / "params" / "intent.json").read_text(encoding="utf-8"))
        ckpts = dict(list_checkpoints(run_dir / "nn"))
        checkpoint = ckpts[max(ckpts)]
        os.environ["UNIFIED_EXTRA_OBS"] = "1"
    else:
        checkpoint = Path(args_cli.unified_checkpoint).expanduser().resolve()

    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 5.0
    env_cfg.scene.clone_in_fabric = False
    env_cfg.seed = args_cli.seed
    env_cfg.sim.render_interval = 10**9
    env_cfg.sim.render.antialiasing_mode = args_cli.antialiasing

    config = agent_cfg["params"]["config"]
    config["num_actors"] = 1
    config["seed"] = args_cli.seed
    config["device"] = args_cli.device
    config["device_name"] = args_cli.device
    agent_cfg["params"]["seed"] = args_cli.seed

    clip_obs = float(agent_cfg["params"]["env"].get("clip_observations", math.inf))
    clip_actions = float(agent_cfg["params"]["env"].get("clip_actions", math.inf))

    base_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if student_record is not None:
        intent = build_intent(
            student_record["intent"],
            expert_checkpoint=Path(student_record["expert_checkpoint"]) if student_record.get("expert_checkpoint") else None,
            expert_sha256=student_record.get("expert_checkpoint_sha256"),
            pca_basis=Path(student_record["pca_basis"]) if student_record.get("pca_basis") else None,
            obj_lookahead=student_record.get("obj_lookahead_rad"),
            unified=True,
        ).to(config["device"])
        env = IntentRlGamesVecEnvWrapper(
            base_env, config["device"], clip_obs, clip_actions, intent,
            goal_hidden=True, asymmetric_critic=True,
            intent_hold=int(student_record.get("intent_hold", 1)),
            intent_delay=int(student_record.get("intent_delay", 0) or 0),
            action_hold=int(student_record.get("action_hold", 1) or 1),
            hinge_angle=True, unified=True,
        )
    else:
        env = RlGamesVecEnvWrapper(base_env, config["device"], clip_obs, clip_actions)

    vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    runner = Runner()
    runner.load(agent_cfg)
    player = runner.create_player()
    player.restore(str(checkpoint))
    player.reset()
    player.has_batch_dimension = True

    u = base_env.unwrapped
    u.max_phase_steps = 0
    u.timeout_terminates = False
    u.pose_z1_cameras()
    u.goal_markers.set_visibility(False)
    u.recording = True

    obs = env.reset()
    for _ in range(15):
        u.sim.render()

    cam_name = "high" if "high" in u.z1_sensors else next(iter(u.z1_sensors))
    mp4_path = out_dir / f"{args_cli.name}_{cam_name}.mp4"
    writer = imageio.get_writer(str(mp4_path), fps=60, codec="libx264", quality=8, macro_block_size=None)

    phases_per_step = []
    turns_per_step = []
    kinds_per_step = []
    total_turns = 0
    prev_completions = 0

    with torch.no_grad():
        for step_i in range(args_cli.steps):
            policy_obs = obs["obs"] if isinstance(obs, dict) else obs
            act = player.get_action(player.obs_to_torch(policy_obs), is_deterministic=True)
            obs, _, _, _ = env.step(act)

            u.sim.render()
            u.z1_sensors[cam_name].update(0.0, force_recompute=True)
            rgb = u.z1_sensors[cam_name].data.output["rgb"][0, ..., :3].detach().cpu().numpy()
            writer.append_data(rgb)

            n_comp = len(u.completions)
            if n_comp > prev_completions:
                for c in u.completions[prev_completions:n_comp]:
                    for row in c:
                        if int(row[0].item()) == TURN_KIND:
                            total_turns += 1
                prev_completions = n_comp

            phases_per_step.append(int(sum(len(c) for c in u.completions)))
            turns_per_step.append(int(total_turns))
            kinds_per_step.append(PHASE_KINDS[int(u.kind[0].item())])

    writer.close()
    falls = int(sum(int(t[:, 3].sum().item()) for t in u.terminals)) if u.terminals else 0
    trace = {
        "name": args_cli.name,
        "checkpoint": str(checkpoint),
        "student_intent": student_record["intent"] if student_record else "expert",
        "intent_hold": int(student_record.get("intent_hold", 1)) if student_record else 1,
        "steps": args_cli.steps,
        "fps": 60,
        "total_phases": phases_per_step[-1] if phases_per_step else 0,
        "total_turns": total_turns,
        "falls": falls,
        "video": mp4_path.name,
        "phases_per_step": phases_per_step,
        "turns_per_step": turns_per_step,
        "kinds_per_step": kinds_per_step,
    }
    (out_dir / f"{args_cli.name}_trace.json").write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    print(f"task3_demo_clip name={args_cli.name} phases={trace['total_phases']} turns={total_turns} falls={falls}", flush=True)
    print("record_task3_unified_demo_status=PASS", flush=True)


if __name__ == "__main__":
    try:
        main()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except Exception:
        traceback.print_exc()
        os._exit(1)
