"""Smoke-check the M2 task (single-axis goals about world z) before training its expert.

Checks: goals and initial cube orientations are pure yaw and span [-pi, pi); the interface (157-D
observations, 20-D actions) matches Stage 1; a cube placed flat and held by fixed joint targets stays
flat, i.e. world z is the palm normal; resampled goals stay yaw-only; random actions give finite rewards.
"""

import argparse
import json
import math

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--hold_steps", type=int, default=60, help="Steps with fixed joint targets (60 = 1 s)")
parser.add_argument("--random_steps", type=int, default=200)
parser.add_argument("--max_settled_tilt_deg", type=float, default=10.0)
parser.add_argument("--zero_joint_noise", action="store_true", help="Reset joints exactly to the default pose")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import m2_tasks
from isaaclab.utils.math import quat_apply, quat_mul, quat_conjugate
from isaaclab_tasks.utils import parse_env_cfg


def yaw_report(name: str, quat: torch.Tensor) -> dict:
    off_axis = float(quat[:, 1:3].abs().max())
    yaw = 2.0 * torch.atan2(quat[:, 3], quat[:, 0])
    yaw = torch.remainder(yaw + math.pi, 2 * math.pi) - math.pi
    report = {"max_abs_qx_qy": off_axis, "yaw_min": float(yaw.min()), "yaw_max": float(yaw.max()),
              "mean_abs_yaw": float(yaw.abs().mean())}
    print(f"{name} {json.dumps(report)}", flush=True)
    return report


def tilt_deg(quat: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(quat[:, 1:4])
    z[:, 2] = 1.0
    return torch.rad2deg(torch.acos(torch.clamp(quat_apply(quat, z)[:, 2], -1.0, 1.0)))


env = None
failures = []
try:
    env_cfg = parse_env_cfg(m2_tasks.M2_TASK, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = 0
    if args.zero_joint_noise:
        env_cfg.reset_dof_pos_noise = 0.0
        env_cfg.reset_dof_vel_noise = 0.0
    print(f"reset_dof_pos_noise={env_cfg.reset_dof_pos_noise}", flush=True)
    env = gym.make(m2_tasks.M2_TASK, cfg=env_cfg)
    core = env.unwrapped
    obs, _ = env.reset()
    print(f"task={m2_tasks.M2_TASK} env_class={type(core).__name__}", flush=True)
    print(f"observation_dim={obs['policy'].shape[-1]} action_dim={env.action_space.shape[-1]}", flush=True)
    if obs["policy"].shape[-1] != 157 or env.action_space.shape[-1] != 20:
        failures.append("interface differs from Stage 1")

    goal = yaw_report("reset_goal", core.goal_rot)
    start = yaw_report("reset_object", core.object_rot)
    for name, report in (("goal", goal), ("object", start)):
        if report["max_abs_qx_qy"] > 1e-5:
            failures.append(f"{name} orientation is not pure yaw")
        if report["yaw_min"] > -2.8 or report["yaw_max"] < 2.8:
            failures.append(f"{name} yaw does not span [-pi, pi)")

    palm = [i for i, body in enumerate(core.hand.body_names) if "palm" in body.lower()]
    if palm:
        normals = -quat_apply(core.hand.data.body_quat_w[:, palm[0]],
                              torch.eye(3, device=core.device)[1:2].expand(core.num_envs, 3))
        print(f"palm_normal_minus_local_y_mean={normals.mean(0).tolist()} "
              f"std={normals.std(0).tolist()}", flush=True)
        palm_quat = core.hand.data.body_quat_w[:1, palm[0]]
        axes = {axis: quat_apply(palm_quat, torch.eye(3, device=core.device)[i : i + 1])[0].tolist()
                for i, axis in enumerate("xyz")}
        print(f"palm_body={core.hand.body_names[palm[0]]} palm_quat_w={palm_quat[0].tolist()} "
              f"palm_local_axes_in_world={json.dumps(axes)}", flush=True)

    idx = core.actuated_dof_indices
    lower, upper = core.hand_dof_lower_limits[:, idx], core.hand_dof_upper_limits[:, idx]
    hold = (2.0 * core.cur_targets[:, idx] - upper - lower) / (upper - lower)
    initial_yaw = core.object_rot.clone()
    ever_reset = torch.zeros(core.num_envs, dtype=torch.bool, device=core.device)
    with torch.inference_mode():
        for _ in range(args.hold_steps):
            _, _, terminated, truncated, _ = env.step(hold)
            ever_reset |= terminated | truncated
        kept = ~ever_reset
        tilt = tilt_deg(core.object_rot[kept])
        drift = torch.rad2deg(2.0 * torch.asin(torch.clamp(
            quat_mul(core.object_rot[kept], quat_conjugate(initial_yaw[kept]))[:, 1:4].norm(dim=-1), max=1.0)))
        settled = {
            "hold_steps": args.hold_steps,
            "fell_fraction": float(ever_reset.float().mean()),
            "tilt_deg_median": float(tilt.median()),
            "tilt_deg_p95": float(tilt.quantile(0.95)),
            "tilt_deg_max": float(tilt.max()),
            "rotation_from_start_deg_median": float(drift.median()),
        }
        print(f"held_flat_cube {json.dumps(settled)}", flush=True)
        if settled["tilt_deg_median"] > args.max_settled_tilt_deg:
            failures.append("cube does not stay flat: world z may not be the palm normal")

        ids = torch.arange(0, core.num_envs, 2, device=core.device)
        core._reset_target_pose(ids)
        resampled = yaw_report("resampled_goal", core.goal_rot[ids])
        if resampled["max_abs_qx_qy"] > 1e-5:
            failures.append("resampled goal is not pure yaw")

        finite = True
        for _ in range(args.random_steps):
            actions = 2.0 * torch.rand(env.action_space.shape, device=core.device) - 1.0
            _, rewards, _, _, _ = env.step(actions)
            finite &= bool(torch.isfinite(rewards).all())
        print(f"random_steps={args.random_steps} rewards_finite={finite} "
              f"goal_still_yaw_only={float(core.goal_rot[:, 1:3].abs().max()) <= 1e-5}", flush=True)
        if not finite:
            failures.append("non-finite reward")
        if float(core.goal_rot[:, 1:3].abs().max()) > 1e-5:
            failures.append("goal left yaw-only during random play")

    for failure in failures:
        print(f"check_failure={failure}", flush=True)
    print(f"m2_task_check_status={'FAIL' if failures else 'PASS'}", flush=True)
finally:
    if env is not None:
        env.close()
    simulation_app.close()
