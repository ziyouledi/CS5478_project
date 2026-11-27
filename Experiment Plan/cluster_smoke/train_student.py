#!/usr/bin/env python3
"""Train an RL-Games PPO student on Shadow Hand with an intent vector appended to its observation.

Adapted from Isaac Lab v2.3.1 scripts/reinforcement_learning/rl_games/train.py. The environment,
reward, PPO configuration and network are unchanged; only the policy observation grows from
157-D to 157 + intent_dim. The intent is computed from the clipped 157-D observation of the
student's own state (see intent_modules.py).

Stage 2 M1 adds two flags (see intent_env_wrapper.py): ``--goal_hidden`` gives the actor the
149-D goal-free observation plus the intent, and ``--asymmetric_critic`` adds a central value net
on the full clipped 157-D observation.

Rate sweep: ``--intent_hold N`` refreshes the intent only every N episode steps (zero-order hold,
see intent_env_wrapper.py); it is written to params/intent.json and the evaluator reads it back.

Rate sweep phase 2: ``--checkpoint`` resumes PPO from a periodic RL-Games checkpoint of a run with the same
intent, hold, seed, task and flags (checked against that run's params/intent.json). RL-Games restores the
actor-critic, input normalizers, central value net, optimizer state, epoch and frame count; it does not restore
the adaptive learning rate, which restarts at the configured value (Isaac Lab's train.py behaves the same).
Training continues to ``--max_iterations`` (absolute epoch count).

Unified task (FACE_TURN_PLAN.md E3): with ``--task Isaac-Unified-HingedCube-Shadow-Direct-v0`` the student runs on
the v4 teacher's training env (settings from the UNIFIED_* / HINGE_* environment variables, recorded in
intent.json); it needs --goal_hidden --asymmetric_critic --hinge_angle_obs (see intent_env_wrapper.py).
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
parser.add_argument("--intent", required=True,
                    help="blind | hand-K | joint-K | wrist-2 | wrist2-finger-K | obj-1|3|4 | goal-4 | goal-yaw | goal-7 | goal-11")
parser.add_argument("--expert_checkpoint", default=None)
parser.add_argument("--expert_sha256", default=None)
parser.add_argument("--pca_basis", default=None)
parser.add_argument("--obj_lookahead", type=float, default=None, help="Subgoal lookahead angle in radians.")
parser.add_argument("--goal_hidden", action="store_true", help="Stage 2 M1: remove the goal fields from the actor input.")
parser.add_argument(
    "--asymmetric_critic",
    action="store_true",
    help="Add an RL-Games central value net that sees the full clipped 157-D observation (no intent).",
)
parser.add_argument("--hinge_angle_obs", action="store_true",
                    help="Face-turn task: give the actor (sin, cos) of the current hinge angle (see intent_env_wrapper.py).")
parser.add_argument("--intent_hold", type=int, default=1, help="Refresh the intent every N episode steps (1 = every step).")
parser.add_argument("--intent_delay", type=int, default=0,
                    help="Actor sees the intent computed this many control steps earlier in the episode (0 = off).")
parser.add_argument("--action_hold", type=int, default=1,
                    help="Apply a new policy action only every N episode steps and repeat it in between (1 = off).")
parser.add_argument("--checkpoint", default=None, help="Resume from this periodic RL-Games checkpoint (see module doc).")
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
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rl_games import RlGamesGpuEnv
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import m2_tasks  # noqa: F401  registers Isaac-Repose-Cube-Shadow-ZAxis-Direct-v0
import hinged_cube_tasks  # noqa: F401  registers the face-turn task
from intent_env_wrapper import IntentRlGamesVecEnvWrapper
from intent_modules import build_intent, sha256
from student_eval_utils import checkpoint_iteration


# Stage 2 M1 critic, copied from Isaac Lab's shadow_hand/agents/rl_games_ppo_ff_cfg.yaml except
# mini_epochs 5 (FF uses 4) to match the Stage 1 actor, whose PPO settings stay unchanged.
CENTRAL_VALUE_CONFIG = {
    "minibatch_size": 32768,
    "mini_epochs": 5,
    "learning_rate": 5e-4,
    "lr_schedule": "adaptive",
    "schedule_type": "standard",
    "kl_threshold": 0.016,
    "clip_value": True,
    "normalize_input": True,
    "truncate_grads": True,
    "network": {
        "name": "actor_critic",
        "central_value": True,
        "mlp": {
            "units": [512, 512, 256, 128],
            "activation": "elu",
            "d2rl": False,
            "initializer": {"name": "default"},
            "regularizer": {"name": "None"},
        },
    },
}


# Fields of the source run's intent.json that a resumed run must share (num_envs may differ in dry runs).
RESUME_MATCH_KEYS = (
    "intent", "task", "seed", "goal_hidden", "asymmetric_critic", "intent_hold", "intent_delay", "action_hold",
    "expert_checkpoint_sha256", "pca_basis_sha256", "obj_lookahead_rad", "hinge_angle_obs",
    "unified_start_states_sha256",
)

# Unified-task settings read by hinged_cube_env.py from the environment (defaults = the v4 teacher's run).
UNIFIED_ENV_DEFAULTS = {
    "HINGE_EFFORT_LIMIT": "0.1", "FACE_TURN_HOLD_STEPS": "10", "UNIFIED_TURN_ORIENT_TOL": "0.4",
    "UNIFIED_MAX_PHASE_STEPS": "120", "UNIFIED_TIMEOUT_TERMINATES": "1", "UNIFIED_HINGE_PROG_SCALE": "30.0",
    "UNIFIED_ORIENT_PROG_SCALE": "15.0", "UNIFIED_HOLD_PROG_SCALE": "5.0", "UNIFIED_ALIVE_BONUS": "0.5",
    "UNIFIED_FALL_PENALTY": "-15.0", "UNIFIED_EXTRA_OBS": "1", "UNIFIED_START_STATES": "",
    "UNIFIED_START_FRACTION": "0.5", "UNIFIED_CYCLE": "",
}


def check_resume_source(checkpoint: Path, record: dict) -> dict:
    """Returns resume provenance; raises if the checkpoint's run differs from this one."""
    start_epoch = checkpoint_iteration(checkpoint)
    if not checkpoint.is_file() or start_epoch is None:
        raise ValueError(f"Resume checkpoint {checkpoint} is not a periodic RL-Games checkpoint (_ep_<N>_)")
    if start_epoch >= record["max_iterations"]:
        raise ValueError(f"Resume epoch {start_epoch} >= --max_iterations {record['max_iterations']}")
    source_path = checkpoint.parent.parent / "params" / "intent.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    defaults = {"intent_hold": 1, "intent_delay": 0, "action_hold": 1, "goal_hidden": False, "asymmetric_critic": False,
                "hinge_angle_obs": False}
    mismatched = {
        key: (source.get(key, defaults.get(key)), record.get(key, defaults.get(key)))
        for key in RESUME_MATCH_KEYS
        if source.get(key, defaults.get(key)) != record.get(key, defaults.get(key))
    }
    if mismatched:
        raise ValueError(f"Resume source {source_path} differs from this run (source, this): {mismatched}")
    return {
        "resume_from": str(checkpoint),
        "resume_from_sha256": sha256(checkpoint),
        "resume_start_epoch": start_epoch,
        "resume_source_num_envs": source.get("num_envs"),
        "resume_source_max_iterations": source.get("max_iterations"),
    }


def max_abs_state_diff(live: dict, saved: dict) -> float:
    if live.keys() != saved.keys():
        raise KeyError(f"State keys differ: {sorted(set(live) ^ set(saved))[:5]}")
    return max(float((live[key].detach().cpu().float() - saved[key].cpu().float()).abs().max()) for key in live)


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
    if args_cli.asymmetric_critic:
        # Same minibatch as the actor, so a minibatch_size override (dry run) applies to both.
        config["central_value_config"] = {**CENTRAL_VALUE_CONFIG, "minibatch_size": config["minibatch_size"]}
    rollout_batch = args_cli.num_envs * int(config["horizon_length"])
    if rollout_batch % int(config["minibatch_size"]) != 0:
        raise ValueError(f"num_envs * horizon_length = {rollout_batch} is not divisible by minibatch_size")

    rl_device = config["device"]
    unified = args_cli.task == hinged_cube_tasks.UNIFIED_TASK
    intent = build_intent(
        args_cli.intent,
        expert_checkpoint=Path(args_cli.expert_checkpoint) if args_cli.expert_checkpoint else None,
        expert_sha256=args_cli.expert_sha256,
        pca_basis=Path(args_cli.pca_basis) if args_cli.pca_basis else None,
        obj_lookahead=args_cli.obj_lookahead,
        unified=unified,
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
        "goal_hidden": args_cli.goal_hidden,
        "asymmetric_critic": args_cli.asymmetric_critic,
        "intent_hold": args_cli.intent_hold,
        "intent_delay": args_cli.intent_delay,
        "action_hold": args_cli.action_hold,
        "hinge_angle_obs": args_cli.hinge_angle_obs,
    }
    if args_cli.task == hinged_cube_tasks.FACE_TURN_TASK:
        # Task settings read by hinged_cube_env.py from the environment; the evaluator checks them.
        intent_record["hinge_effort_limit"] = float(os.environ.get("HINGE_EFFORT_LIMIT", "0.1"))
        intent_record["face_turn_hold_steps"] = int(os.environ.get("FACE_TURN_HOLD_STEPS", "1"))
        intent_record["face_turn_start_states"] = os.environ.get("FACE_TURN_START_STATES") or None
    elif unified:
        settings = {key: os.environ.get(key, default) for key, default in UNIFIED_ENV_DEFAULTS.items()}
        diag = os.environ.get("E3_DIAG", "")
        if settings["UNIFIED_EXTRA_OBS"] == "0" or (settings["UNIFIED_CYCLE"] and diag != "anyonly"):
            raise ValueError("Unified students train on the v4 env: UNIFIED_EXTRA_OBS=1, random goal order "
                             "(a fixed order only for the E3_DIAG=anyonly diagnostic)")
        intent_record["unified_env"] = settings
        intent_record["e3_diag"] = diag or None
        start = settings["UNIFIED_START_STATES"]
        intent_record["unified_start_states_sha256"] = sha256(Path(start).expanduser()) if start else None
    elif args_cli.hinge_angle_obs:
        raise ValueError("--hinge_angle_obs is only defined on the face-turn and unified tasks")
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve() if args_cli.checkpoint else None
    if checkpoint is not None:
        intent_record.update(check_resume_source(checkpoint, intent_record))

    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = IntentRlGamesVecEnvWrapper(
        env,
        rl_device,
        clip_obs,
        clip_actions,
        intent,
        goal_hidden=args_cli.goal_hidden,
        asymmetric_critic=args_cli.asymmetric_critic,
        intent_hold=args_cli.intent_hold,
        intent_delay=args_cli.intent_delay,
        action_hold=args_cli.action_hold,
        hinge_angle=args_cli.hinge_angle_obs,
        unified=unified,
    )
    intent_record["student_obs_dim"] = env.actor_obs_dim
    intent_record["critic_states_dim"] = env.rlg_num_states
    (log_dir / "params" / "intent.json").write_text(json.dumps(intent_record, indent=2) + "\n", encoding="utf-8")
    for key, value in intent_record.items():
        print(f"student_{key}={value}")

    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
    config["num_actors"] = env.unwrapped.num_envs

    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)
    runner.reset()
    if checkpoint is None:
        runner.run({"train": True, "play": False, "sigma": None})
    else:
        # Same steps as Runner.run_train with a checkpoint (sigma override is None), plus a check that
        # the restored networks equal the checkpoint and training resumes at its epoch.
        agent = runner.algo_factory.create(runner.algo_name, base_name="run", params=runner.params)
        agent.restore(str(checkpoint))
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model_diff = max_abs_state_diff(agent.model.state_dict(), saved["model"])
        critic_diff = (
            max_abs_state_diff(agent.central_value_net.state_dict(), saved["assymetric_vf_nets"])
            if agent.has_central_value else 0.0
        )
        print(
            f"resume_start_epoch={agent.epoch_num} resume_frame={agent.frame} "
            f"resume_model_max_abs_diff={model_diff:.3e} resume_central_value_max_abs_diff={critic_diff:.3e} "
            f"resume_has_central_value={agent.has_central_value}"
        )
        if agent.epoch_num != intent_record["resume_start_epoch"] or model_diff != 0.0 or critic_diff != 0.0:
            raise RuntimeError("Resumed agent does not match its checkpoint")
        if agent.has_central_value != args_cli.asymmetric_critic:
            raise RuntimeError("Central value net presence does not match --asymmetric_critic")
        agent.train()
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
