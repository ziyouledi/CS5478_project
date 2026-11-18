#!/usr/bin/env python3
"""Run two frozen experts on the task-switching env (hinged_cube_env.py: ShadowHandSwitchEnv).

Each env follows SWITCH_CYCLE; at every control step an env in a ``turn`` phase is driven by the
face-turn expert and any other env by the Stage 1 (reorientation) expert, both deterministic. Runs
``--steps`` control steps from a reset and writes per-phase statistics: how many phases are completed,
how long a phase takes depending on which phase came before it, falls, and the cube state at handoff.
No training; nothing here is a reward used for learning.

``--selector_hold N`` / ``--selector_delay d`` (two experts only, FACE_TURN_PLAN.md E2'): the experts see
the selector message (phase kind, goal) through an IntentChannel; the kind in the message picks the expert
and the goal fields obs[64:72] are rebuilt from the message goal. The env still switches goals and tests
success on the true current phase, so after a completed goal a stale message keeps the old skill on the old
goal. Where the message equals the current (kind, goal) the env's own fields are kept, so the defaults
(N = 1, d = 0) leave the observations bit-identical (fraction_env_steps_fields_replaced = 0).

``--student_run_dir`` (FACE_TURN_PLAN.md E3): one goal-hidden student of the v4 unified teacher on the unified
task, evaluated like ``--unified_checkpoint`` (phase timeout off, stock resets, fixed UNIFIED_CYCLE). The env is
wrapped exactly as in training (intent_env_wrapper.py, unified=True) from the run's params/intent.json: same
intent, intent hold / delay, PCA basis and teacher; the actor sees unified_actor_obs + intent. Default checkpoint
= the last periodic one in <run_dir>/nn.
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

import hinged_cube_tasks  # noqa: F401

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=hinged_cube_tasks.SWITCH_TASK)
parser.add_argument("--agent", default="rl_games_cfg_entry_point")
parser.add_argument("--reorient_checkpoint", default=None)
parser.add_argument("--turn_checkpoint", default=None)
parser.add_argument("--unified_checkpoint", default=None,
                    help="One checkpoint of the unified task (ShadowHandUnifiedEnv) instead of the two experts; "
                         "the goal order comes from UNIFIED_CYCLE (unset = the random order used in training)")
parser.add_argument("--student_run_dir", default=None,
                    help="Unified-task student run (contains nn/ and params/intent.json), see module doc")
parser.add_argument("--student_checkpoint", default=None,
                    help="With --student_run_dir: this checkpoint instead of the last periodic one")
parser.add_argument("--student_iteration", type=int, default=None,
                    help="With --student_run_dir: the periodic checkpoint of this iteration instead of the last one")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--seed", type=int, default=20261006)
parser.add_argument("--stall_steps", type=int, default=120, help="A phase still open after this many steps counts as stalled")
parser.add_argument("--output", required=True)
parser.add_argument("--capture", default=None, help="Comma-separated phase kinds whose end states are saved")
parser.add_argument("--capture_output", default=None, help="npz for the captured states (FACE_TURN_START_STATES)")
parser.add_argument("--selector_hold", type=int, default=1,
                    help="Two experts only: the selector message (skill kind, goal) is refreshed when the episode "
                         "step is a multiple of N and held otherwise (IntentChannel hold)")
parser.add_argument("--selector_delay", type=int, default=0,
                    help="Two experts only: the policies see the selector message of d steps earlier in the episode")
parser.add_argument("--only_expert", choices=("reorient", "turn"), default=None,
                    help="Control: drive every phase with this one expert (its own observation layout for every phase)")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
single_policy = bool(args_cli.unified_checkpoint or args_cli.student_run_dir)
if args_cli.unified_checkpoint and args_cli.student_run_dir:
    parser.error("pass --unified_checkpoint or --student_run_dir, not both")
if args_cli.student_checkpoint and not args_cli.student_run_dir:
    parser.error("--student_checkpoint needs --student_run_dir")
if args_cli.student_iteration is not None and (not args_cli.student_run_dir or args_cli.student_checkpoint):
    parser.error("--student_iteration needs --student_run_dir and excludes --student_checkpoint")
if single_policy and (args_cli.selector_hold != 1 or args_cli.selector_delay != 0 or args_cli.only_expert):
    parser.error("--selector_hold / --selector_delay / --only_expert apply to the two-expert path only")
if single_policy and (args_cli.reorient_checkpoint or args_cli.turn_checkpoint):
    parser.error("--reorient_checkpoint / --turn_checkpoint apply to the two-expert path only")
if args_cli.only_expert and (args_cli.selector_hold != 1 or args_cli.selector_delay != 0):
    parser.error("--only_expert is a control at N = 1, d = 0")
if single_policy:
    args_cli.task = hinged_cube_tasks.UNIFIED_TASK
elif not (args_cli.reorient_checkpoint and args_cli.turn_checkpoint):
    parser.error("pass --unified_checkpoint, or both --reorient_checkpoint and --turn_checkpoint")
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

from isaaclab.utils.math import quat_conjugate, quat_mul

from hinged_cube_env import PHASE_KINDS, TURN_KIND
from intent_env_wrapper import IntentRlGamesVecEnvWrapper
from intent_modules import GOAL_FIELDS, IntentChannel, build_intent, canonical_quat, selector_view, sha256
from student_eval_utils import list_checkpoints

# Unified-env settings that change what a student sees or what counts as success; the evaluation must use the
# training values from intent.json. Phase timeout and start states are evaluation-protocol settings (see
# _student_setup), as for the teachers.
STUDENT_MATCH_ENV = ("HINGE_EFFORT_LIMIT", "FACE_TURN_HOLD_STEPS", "UNIFIED_TURN_ORIENT_TOL", "UNIFIED_EXTRA_OBS")
STUDENT_ENV_DEFAULTS = {"HINGE_EFFORT_LIMIT": "0.1", "FACE_TURN_HOLD_STEPS": "10", "UNIFIED_TURN_ORIENT_TOL": "0.4",
                        "UNIFIED_EXTRA_OBS": "1"}


def _student_setup(run_dir: Path, checkpoint: str | None, iteration: int | None = None) -> tuple[Path, dict]:
    """Checkpoint and intent.json of a unified-task student; checks the env settings against training."""
    record_path = run_dir / "params" / "intent.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("task") != hinged_cube_tasks.UNIFIED_TASK:
        raise ValueError(f"{record_path}: task {record.get('task')} is not the unified task")
    if not (record.get("goal_hidden") and record.get("asymmetric_critic") and record.get("hinge_angle_obs")):
        raise ValueError(f"{record_path}: unified students are goal-hidden with asymmetric critic and hinge angle")
    trained = record.get("unified_env") or {}
    for key in STUDENT_MATCH_ENV:
        current = os.environ.get(key, STUDENT_ENV_DEFAULTS[key])
        if key not in trained or float(trained[key]) != float(current):
            raise ValueError(f"{key}={current} differs from the training value {trained.get(key)} in {record_path}")
    if os.environ.get("UNIFIED_START_STATES", "").strip():
        raise ValueError("UNIFIED_START_STATES must be unset for evaluation (stock resets, as for the teachers)")
    if os.environ.get("UNIFIED_MAX_PHASE_STEPS", "0") != "0":
        raise ValueError("UNIFIED_MAX_PHASE_STEPS must be 0 (or unset) for evaluation, as for the teachers")
    os.environ["UNIFIED_MAX_PHASE_STEPS"] = "0"
    pca_basis = record.get("pca_basis")
    if pca_basis and record.get("pca_basis_sha256") and sha256(Path(pca_basis)) != record["pca_basis_sha256"]:
        raise ValueError(f"PCA basis {pca_basis} changed since training")
    if checkpoint:
        path = Path(checkpoint).expanduser().resolve()
    else:
        found = list_checkpoints(run_dir / "nn", 1)
        if not found:
            raise FileNotFoundError(f"No periodic checkpoints in {run_dir / 'nn'}")
        if iteration is not None:
            by_iteration = dict(found)
            if iteration not in by_iteration:
                raise FileNotFoundError(f"No periodic checkpoint of iteration {iteration} in {run_dir / 'nn'}")
            path = Path(by_iteration[iteration])
        else:
            last, path = found[-1]
            if last != record.get("max_iterations"):
                raise ValueError(f"Last periodic checkpoint is iteration {last}, run planned {record.get('max_iterations')}")
            path = Path(path)
    return path, record

# Per-step trace of the unified task (one row per env and step, taken after the step).
TRACE_COLUMNS = (
    "kind", "phase_steps", "hinge_error_rad", "orient_error_rad", "tilt_error_rad", "yaw_error_rad",
    "hold_count", "hinge_pos_rad", "phase_start_hinge_rad", "reward",
)


def _unified_trace_row(unwrapped, reward: torch.Tensor) -> torch.Tensor:
    """State of every env of ShadowHandUnifiedEnv against its current goal.

    orient_error is the rotation distance of the success test (cube = bottom part against goal_rot);
    tilt_error is the angle between the cube's z axis and the goal's, yaw_error the twist about z of
    the same error rotation (world frame)."""
    q = quat_mul(unwrapped.object_rot, quat_conjugate(unwrapped.goal_rot))
    w, x, y, z = q.unbind(-1)
    orient = 2.0 * torch.asin(torch.clamp(torch.norm(q[:, 1:4], dim=-1), max=1.0))
    tilt = torch.acos(torch.clamp(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))
    yaw = 2.0 * torch.atan2(z, w)
    yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
    return torch.stack(
        (
            unwrapped.kind.float(),
            unwrapped.phase_steps.float(),
            unwrapped._hinge_error(),
            orient,
            tilt,
            yaw,
            unwrapped.hold_count.float(),
            unwrapped.object.data.joint_pos[:, 0],
            unwrapped.phase_start_hinge,
            reward.reshape(-1).to(unwrapped.device).float(),
        ),
        dim=-1,
    ).cpu()


def _open_turn_diagnostics(trace: np.ndarray, stall_steps: int, hinge_tol: float, orient_tol: float) -> dict:
    """What stalled turn phases look like: rows of the trace in a turn phase open for more than stall_steps."""
    col = {name: i for i, name in enumerate(TRACE_COLUMNS)}
    rows = trace.reshape(-1, trace.shape[-1])
    rows = rows[(rows[:, col["kind"]] == TURN_KIND) & (rows[:, col["phase_steps"]] > stall_steps)]
    if len(rows) == 0:
        return {"n": 0}
    hinge = np.abs(rows[:, col["hinge_error_rad"]])
    orient = rows[:, col["orient_error_rad"]]
    tilt = rows[:, col["tilt_error_rad"]]
    yaw = np.abs(rows[:, col["yaw_error_rad"]])
    moved = np.abs(rows[:, col["hinge_pos_rad"]] - rows[:, col["phase_start_hinge_rad"]])
    at_goal = hinge <= hinge_tol
    orient_ok = orient <= orient_tol
    quarter = math.pi / 2.0

    def frac(mask) -> float:
        return float(mask.mean())

    def deg(values) -> dict:
        return _stats(np.rad2deg(values))

    return {
        "n": int(len(rows)),
        "fraction_hinge_at_goal": frac(at_goal),
        "fraction_hinge_at_goal_orient_outside": frac(at_goal & ~orient_ok),
        "fraction_hinge_at_goal_orient_inside": frac(at_goal & orient_ok),
        "fraction_hinge_not_at_goal_orient_inside": frac(~at_goal & orient_ok),
        "fraction_hinge_not_at_goal_orient_outside": frac(~at_goal & ~orient_ok),
        "fraction_hinge_within_tol_of_start": frac(moved <= hinge_tol),
        "fraction_hinge_error_45_to_135_deg": frac((hinge > quarter / 2) & (hinge <= 1.5 * quarter)),
        "fraction_hinge_error_over_135_deg": frac(hinge > 1.5 * quarter),
        "fraction_hinge_error_tol_to_45_deg": frac((hinge > hinge_tol) & (hinge <= quarter / 2)),
        "hinge_error_deg": deg(hinge),
        "orient_error_deg": deg(orient),
        "tilt_error_deg": deg(tilt),
        "abs_yaw_error_deg": deg(yaw),
        "at_goal_orient_outside": {
            "orient_error_deg": deg(orient[at_goal & ~orient_ok]),
            "tilt_error_deg": deg(tilt[at_goal & ~orient_ok]),
            "abs_yaw_error_deg": deg(yaw[at_goal & ~orient_ok]),
            "fraction_tilt_alone_outside": float((tilt[at_goal & ~orient_ok] > orient_tol).mean())
            if (at_goal & ~orient_ok).any() else None,
            "fraction_yaw_alone_outside": float((yaw[at_goal & ~orient_ok] > orient_tol).mean())
            if (at_goal & ~orient_ok).any() else None,
        },
        "mean_reward_per_step": float(rows[:, col["reward"]].mean()),
        "mean_reward_per_step_hinge_at_goal": float(rows[at_goal, col["reward"]].mean()) if at_goal.any() else None,
        "mean_reward_per_step_hinge_not_at_goal": float(rows[~at_goal, col["reward"]].mean())
        if (~at_goal).any() else None,
    }


def _stats(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {"n": 0}
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
    }


def _kind_name(index: float) -> str:
    return "reset" if index < 0 else PHASE_KINDS[int(index)]


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: dict) -> None:
    student_record = None
    if args_cli.student_run_dir:
        run_dir = Path(args_cli.student_run_dir).expanduser().resolve()
        path, student_record = _student_setup(run_dir, args_cli.student_checkpoint, args_cli.student_iteration)
        checkpoints = {"student": path}
    elif args_cli.unified_checkpoint:
        path = Path(args_cli.unified_checkpoint).expanduser().resolve()
        checkpoints = {"unified": path}
        os.environ.setdefault("UNIFIED_MAX_PHASE_STEPS", "0")
        if "UNIFIED_EXTRA_OBS" not in os.environ and path.is_file():
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            w = ckpt.get("model", {}).get("a2c_network.actor_mlp.0.weight")
            if w is not None and w.shape[1] == 159:
                os.environ["UNIFIED_EXTRA_OBS"] = "0"
    else:
        checkpoints = {
            "reorient": Path(args_cli.reorient_checkpoint).expanduser().resolve(),
            "turn": Path(args_cli.turn_checkpoint).expanduser().resolve(),
        }
    for path in checkpoints.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    config = agent_cfg["params"]["config"]
    config["num_actors"] = args_cli.num_envs
    config["seed"] = args_cli.seed
    config["device"] = args_cli.device
    config["device_name"] = args_cli.device
    agent_cfg["params"]["seed"] = args_cli.seed

    base_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    if student_record is not None:
        record = student_record
        intent = build_intent(
            record["intent"],
            expert_checkpoint=Path(record["expert_checkpoint"]) if record.get("expert_checkpoint") else None,
            expert_sha256=record.get("expert_checkpoint_sha256"),
            pca_basis=Path(record["pca_basis"]) if record.get("pca_basis") else None,
            obj_lookahead=record.get("obj_lookahead_rad"),
            unified=True,
        ).to(config["device"])
        # Same wrapper as training; the critic states it returns are not used here.
        env = IntentRlGamesVecEnvWrapper(
            base_env, config["device"], clip_obs, clip_actions, intent, goal_hidden=True, asymmetric_critic=True,
            intent_hold=int(record.get("intent_hold", 1)), intent_delay=int(record.get("intent_delay", 0) or 0),
            action_hold=int(record.get("action_hold", 1) or 1), hinge_angle=True, unified=True,
        )
        if env.actor_obs_dim != record.get("student_obs_dim"):
            raise ValueError(f"Actor input {env.actor_obs_dim}-D differs from training {record.get('student_obs_dim')}")
    else:
        env = RlGamesVecEnvWrapper(base_env, config["device"], clip_obs, clip_actions)
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    players = {}
    for name, path in checkpoints.items():
        runner = Runner()
        runner.load(agent_cfg)
        player = runner.create_player()
        player.restore(str(path))
        player.reset()
        player.has_batch_dimension = True
        players[name] = player

    unwrapped = base_env.unwrapped
    obs = env.reset()
    unwrapped.recording = True
    if args_cli.capture:
        unwrapped.capture_kinds = {PHASE_KINDS.index(name) for name in args_cli.capture.split(",")}
    kind_steps = torch.zeros(len(PHASE_KINDS), dtype=torch.float64, device=unwrapped.device)
    trace_rows: list[torch.Tensor] = []
    # Selector channel (E2'): the policies see the (kind, goal) message held / delayed; the env itself
    # switches goals and tests success as before. Pass-through (N = 1, d = 0) must equal the env's fields.
    channel = None if single_policy else IntentChannel(hold=args_cli.selector_hold, delay=args_cli.selector_delay)
    stale = {"kind_steps": 0, "goal_steps": 0, "replaced_steps": 0}
    passthrough_max_diff = 0.0
    # goal-7 students: the intent the actor actually received (last 7 actor inputs) against the env's current goal
    # (canonical goal quaternion, sin/cos hinge goal, is_turn), per phase kind. With hold > 1 stale steps differ.
    goal7_check = None
    if student_record is not None and student_record["intent"] == "goal-7":
        goal7_check = {name: {"steps": 0, "max_abs_error": 0.0, "steps_error_over_1e-3": 0,
                              "max_abs_error_quat": 0.0} for name in PHASE_KINDS}
    with torch.inference_mode():
        for _ in range(args_cli.steps):
            policy_obs = obs["obs"] if isinstance(obs, dict) else obs
            kind = unwrapped.current_kind()
            if goal7_check is not None:
                seen = policy_obs[:, -7:].float()
                goal = unwrapped.hinge_goal.to(seen.device).float()
                want = torch.cat([
                    canonical_quat(unwrapped.goal_rot.to(seen.device).float()),
                    torch.sin(goal).unsqueeze(1), torch.cos(goal).unsqueeze(1),
                    (kind.to(seen.device) == TURN_KIND).float().unsqueeze(1),
                ], dim=1)
                err = (seen - want).abs()
                err_all, err_quat = err.max(dim=1).values, err[:, :4].max(dim=1).values
                for index, name in enumerate(PHASE_KINDS):
                    mask = kind.to(seen.device) == index
                    if mask.any():
                        entry = goal7_check[name]
                        entry["steps"] += int(mask.sum())
                        entry["max_abs_error"] = max(entry["max_abs_error"], float(err_all[mask].max()))
                        entry["max_abs_error_quat"] = max(entry["max_abs_error_quat"], float(err_quat[mask].max()))
                        entry["steps_error_over_1e-3"] += int((err_all[mask] > 1e-3).sum())
            is_turn = (kind == TURN_KIND).to(policy_obs.device)
            kind_steps += torch.bincount(kind, minlength=len(PHASE_KINDS)).double()
            if channel is not None:
                message_kind, fields = selector_view(
                    channel, kind, unwrapped.goal_rot, unwrapped.object_rot, unwrapped.hinge_rot,
                    unwrapped.episode_length_buf, TURN_KIND, mul=quat_mul, conjugate=quat_conjugate,
                )
                fields = fields.to(policy_obs.device, policy_obs.dtype)
                if channel.passthrough:
                    diff = (fields - policy_obs[:, GOAL_FIELDS]).abs().max()
                    passthrough_max_diff = max(passthrough_max_diff, float(diff))
                kind_stale = message_kind != kind
                goal_stale = (fields[:, 0:4] != unwrapped.goal_rot.to(fields.device)).any(dim=-1)
                stale["kind_steps"] += int(kind_stale.sum())
                stale["goal_steps"] += int(goal_stale.sum())
                # Where the message equals the env's current (kind, goal), keep the env's own fields: the
                # recomputed ones differ by ~1e-7 (job 1160295), which changed the N = 1 result by 1-2%.
                fresh = ~(kind_stale | goal_stale).to(policy_obs.device)
                fields = torch.where(fresh.unsqueeze(-1), policy_obs[:, GOAL_FIELDS], fields)
                stale["replaced_steps"] += int((~fresh).sum())
                policy_obs = policy_obs.clone()
                policy_obs[:, GOAL_FIELDS] = fields
                is_turn = (message_kind == TURN_KIND).to(policy_obs.device)
                if args_cli.only_expert:
                    # the one expert sees its own layout in every phase: the message kind is replaced by it
                    forced = torch.full_like(kind, TURN_KIND if args_cli.only_expert == "turn" else 0)
                    _, fields = selector_view(
                        IntentChannel(), forced, unwrapped.goal_rot, unwrapped.object_rot, unwrapped.hinge_rot,
                        unwrapped.episode_length_buf, TURN_KIND, mul=quat_mul, conjugate=quat_conjugate,
                    )
                    fields = fields.to(policy_obs.device, policy_obs.dtype)
                    same_layout = ((forced == TURN_KIND) == (kind == TURN_KIND)).to(policy_obs.device)
                    policy_obs[:, GOAL_FIELDS] = torch.where(same_layout.unsqueeze(-1), policy_obs[:, GOAL_FIELDS], fields)
                    is_turn = torch.full_like(is_turn, args_cli.only_expert == "turn")
            if single_policy:
                (player,) = players.values()
                actions = player.get_action(player.obs_to_torch(policy_obs), is_deterministic=True)
            else:
                turn_actions = players["turn"].get_action(
                    players["turn"].obs_to_torch(policy_obs), is_deterministic=True
                )
                reorient_actions = players["reorient"].get_action(
                    players["reorient"].obs_to_torch(policy_obs), is_deterministic=True
                )
                actions = torch.where(is_turn.unsqueeze(-1), turn_actions, reorient_actions)
            obs, step_reward, _, _ = env.step(actions)
            if single_policy:
                trace_rows.append(_unified_trace_row(unwrapped, step_reward))

    env_steps = args_cli.num_envs * args_cli.steps
    per_600 = 600.0 / env_steps
    completions = torch.cat(unwrapped.completions).numpy() if unwrapped.completions else np.zeros((0, 6))
    terminals = torch.cat(unwrapped.terminals).numpy() if unwrapped.terminals else np.zeros((0, 4))
    open_kind = unwrapped.current_kind().cpu().numpy()
    open_steps = unwrapped.phase_steps.cpu().numpy()

    fixed_cycle = bool(unwrapped.cycle_names)
    cycle = unwrapped.cycle_names or ["any", "up", "turn"]  # random order: all three kinds occur
    result = {
        "record_type": "shadow_task_switch_evaluation",
        "cycle": cycle if fixed_cycle else "random",
        "hold_steps": unwrapped.hold_steps,
        "checkpoints": {name: str(path) for name, path in checkpoints.items()},
        "seed": args_cli.seed,
        "num_envs": args_cli.num_envs,
        "steps": args_cli.steps,
        "stall_steps": args_cli.stall_steps,
        "student": None if student_record is None else {
            "run_dir": str(Path(args_cli.student_run_dir).expanduser().resolve()),
            "checkpoint_sha256": sha256(checkpoints["student"]),
            "intent": student_record["intent"],
            "intent_dim": intent.dim,
            "intent_hold": env.intent_hold,
            "intent_delay": env.intent_delay,
            "actor_obs_dim": env.actor_obs_dim,
            "seed": student_record.get("seed"),
            "teacher_sha256": student_record.get("expert_checkpoint_sha256"),
            "pca_basis_sha256": student_record.get("pca_basis_sha256"),
            "unified_goal_check_max_error": env.unified_check_max_error,
            "iteration_requested": args_cli.student_iteration,
            "checkpoint": str(checkpoints["student"]),
            "goal7_intent_check": None if goal7_check is None else {
                name: {**entry, "fraction_steps_error_over_1e-3": entry["steps_error_over_1e-3"] / entry["steps"]}
                for name, entry in goal7_check.items() if entry["steps"]
            },
        },
        "selector": None if channel is None else {
            "hold": args_cli.selector_hold,
            "only_expert": args_cli.only_expert,
            "delay": args_cli.selector_delay,
            "fraction_env_steps_kind_stale": stale["kind_steps"] / env_steps,
            "fraction_env_steps_goal_stale": stale["goal_steps"] / env_steps,
            "fraction_env_steps_fields_replaced": stale["replaced_steps"] / env_steps,
            # recomputed vs env fields before the substitution above (diagnostic; the policy got the env's)
            "passthrough_max_abs_goal_field_diff": passthrough_max_diff if channel.passthrough else None,
        },
        "hinge": unwrapped.hinge_summary(),
        "phases_completed_per_600_steps": float(len(completions) * per_600),
        "cycles_completed_per_600_steps": float(len(completions) * per_600 / len(cycle)) if fixed_cycle else None,
        "falls_per_600_steps": float(terminals[:, 3].sum() * per_600),
        "time_outs": int((terminals[:, 3] == 0).sum()),
        "fraction_of_steps_by_kind": {
            PHASE_KINDS[i]: float(kind_steps[i] / env_steps) for i in range(len(PHASE_KINDS)) if kind_steps[i] > 0
        },
        "by_kind": {},
        "by_transition": {},
    }
    for index, name in enumerate(PHASE_KINDS):
        if name not in cycle:
            continue
        rows = completions[completions[:, 0] == index]
        ended = terminals[terminals[:, 0] == index]
        # phases that never finished: cut by a fall, by the time-out, or still open at the end of the run
        unfinished = np.concatenate([ended[:, 2], open_steps[open_kind == index]])
        entry = {
            "completed_per_600_steps": float(len(rows) * per_600),
            "steps_to_complete": _stats(rows[:, 2]),
            "tilt_deg_at_completion": _stats(rows[:, 3]),
            "falls_per_600_steps": float(ended[:, 3].sum() * per_600),
            "stalled_phases": int((unfinished > args_cli.stall_steps).sum()),
            "stalled_fraction_of_started": float(
                (unfinished > args_cli.stall_steps).sum() / max(len(rows) + len(unfinished), 1)
            ),
        }
        if index != TURN_KIND:
            entry["hinge_drift_deg"] = _stats(rows[:, 4])
            entry["fraction_hinge_moved_over_45_deg"] = float((rows[:, 4] > 45.0).mean()) if len(rows) else None
        else:
            entry["detent_error_deg_at_completion"] = _stats(rows[:, 5])
        result["by_kind"][name] = entry
    for prev in sorted(set(completions[:, 1].tolist())):
        for index in sorted(set(completions[:, 0].tolist())):
            rows = completions[(completions[:, 1] == prev) & (completions[:, 0] == index)]
            if len(rows) == 0:
                continue
            falls = terminals[(terminals[:, 1] == prev) & (terminals[:, 0] == index) & (terminals[:, 3] == 1)]
            result["by_transition"][f"{_kind_name(prev)}->{_kind_name(index)}"] = {
                "steps_to_complete": _stats(rows[:, 2]),
                "tilt_deg_at_completion": _stats(rows[:, 3]),
                "falls": int(len(falls)),
                "fall_fraction_of_started": float(len(falls) / (len(rows) + len(falls))),
            }

    output = Path(args_cli.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if trace_rows:
        trace = torch.stack(trace_rows).numpy().astype(np.float32)  # steps x envs x columns
        result["trace_columns"] = list(TRACE_COLUMNS)
        if "turn" in cycle:
            result["open_turn_diagnostics"] = _open_turn_diagnostics(
                trace, args_cli.stall_steps, float(unwrapped.cfg.success_tolerance), float(unwrapped.turn_orient_tol)
            )
            print(f"open_turn={json.dumps(result['open_turn_diagnostics'])}")
        np.savez_compressed(output.with_name(output.stem + "-trace.npz"), trace=trace, columns=np.array(TRACE_COLUMNS))
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(output.with_suffix(".npz"), completions=completions, terminals=terminals,
                        open_kind=open_kind, open_steps=open_steps)
    if args_cli.capture:
        captured = {
            key: torch.cat([chunk[key] for chunk in unwrapped.captured]).numpy() for key in unwrapped.captured[0]
        }
        capture_path = Path(args_cli.capture_output).expanduser().resolve()
        capture_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(capture_path, **captured)
        print(f"captured_states={len(captured['hinge_pos'])} kinds={args_cli.capture} file={capture_path}")
    print(f"switch_cycle={','.join(cycle) if fixed_cycle else 'random'}")
    if result["student"] is not None:
        print(f"student={json.dumps(result['student'])}")
    if result["selector"] is not None:
        print(f"selector={json.dumps(result['selector'])}")
    for key in ("phases_completed_per_600_steps", "cycles_completed_per_600_steps", "falls_per_600_steps"):
        print(f"{key}={result[key]:.3f}" if result[key] is not None else f"{key}=n/a")
    for name, entry in result["by_kind"].items():
        print(f"kind={name} {json.dumps(entry)}")
    for name, entry in result["by_transition"].items():
        print(f"transition={name} {json.dumps(entry)}")
    print("switch_evaluation_status=PASS")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        print("switch_evaluation_status=FAIL", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    simulation_app.close()
