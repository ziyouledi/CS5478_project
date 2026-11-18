#!/usr/bin/env python3
"""Validate intent modules on recorded expert rollout data (CPU, no Isaac Sim)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from intent_modules import (
    FINGERS,
    GOAL_ROT,
    OBJECT_ROT,
    WRIST,
    build_intent,
    canonical_quat,
    hinge_angle_obs,
    quat_conjugate,
    quat_mul,
    rotation_distance,
    UNIFIED_ACTOR_DIM,
    UNIFIED_EXT_DIM,
    UNIFIED_OBS_DIM,
    unified_actor_obs,
)

UNIFIED_INTENTS = ["blind", "hand-2", "hand-4", "hand-8", "hand-20", "goal-7"]
TURN_KIND = 3  # hinged_cube_env.PHASE_KINDS.index("turn"); that module needs Isaac Sim

STAGE_ONE_INTENTS = ["blind", "wrist-2", "hand-2", "hand-4", "hand-20", "obj-1", "obj-4"]
EXTRA_INTENTS = ["hand-8", "wrist2-finger-2", "obj-3", "goal-4"]
JOINT_INTENTS = ["joint-2", "joint-4", "joint-20"]
EXPERT_TOLERANCE = 1e-4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--pca_basis", required=True)
    parser.add_argument("--expert_checkpoint", required=True)
    parser.add_argument("--expert_sha256", required=True)
    parser.add_argument("--obj_lookahead", type=float, required=True)
    parser.add_argument("--joint_basis", default=None, help="joint_basis.npz from fit_joint_basis.py; adds joint-K checks")
    parser.add_argument("--check_goal_yaw", action="store_true",
                        help="M2 rollout only: goals must be pure yaw; adds goal-yaw and its reconstruction checks")
    parser.add_argument("--face_turn", action="store_true",
                        help="Face-turn rollout: obs[68:72] is the hinge error, so the object-intent checks are "
                             "skipped; adds goal-yaw and hinge-angle checks")
    parser.add_argument("--unified", action="store_true",
                        help="Unified v4 rollout (E3): only blind / hand-K / goal-7, on the 164-D extended observation")
    parser.add_argument("--num_rows", type=int, default=131072)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    observations = np.load(dataset_dir / "observations.npy", mmap_mode="r")
    actions = np.load(dataset_dir / "actions.npy", mmap_mode="r")
    rng = np.random.default_rng(args.seed)
    rows = np.sort(rng.choice(observations.shape[0], size=min(args.num_rows, observations.shape[0]), replace=False))
    obs = torch.from_numpy(np.asarray(observations[rows], dtype=np.float32))
    stored_action = torch.from_numpy(np.asarray(actions[rows], dtype=np.float32))
    basis = np.load(args.pca_basis)
    report: dict = {"dataset_dir": str(dataset_dir), "num_rows": int(len(rows)), "sample_seed": args.seed}
    failures: list[str] = []

    def check(name: str, value: float, limit: float) -> None:
        report.setdefault("checks", {})[name] = {"value": value, "limit": limit, "pass": bool(value <= limit)}
        print(f"check {name} value={value:.3e} limit={limit:.1e} {'PASS' if value <= limit else 'FAIL'}")
        if not value <= limit:
            failures.append(name)

    common = dict(
        expert_checkpoint=Path(args.expert_checkpoint),
        expert_sha256=args.expert_sha256,
        pca_basis=Path(args.pca_basis),
        obj_lookahead=args.obj_lookahead,
    )
    if args.unified:
        validate_unified(args, dataset_dir, rows, obs, stored_action, basis, common, report, failures, check)
        return
    modules = {name: build_intent(name, **common) for name in STAGE_ONE_INTENTS + EXTRA_INTENTS}
    if args.check_goal_yaw or args.face_turn:
        modules["goal-yaw"] = build_intent("goal-yaw", **common)
    if args.joint_basis:
        joint_common = {**common, "pca_basis": Path(args.joint_basis)}
        modules.update({name: build_intent(name, **joint_common) for name in JOINT_INTENTS})
    expert = modules["hand-20"].expert
    report["expert_layer_sizes"] = expert.layer_sizes

    # 1. The rebuilt expert reproduces the recorded RL-Games player actions.
    expert_action = expert(obs)
    check("expert_vs_recorded_action_max_abs", float((expert_action - stored_action).abs().max()), EXPERT_TOLERANCE)

    # 2. Hand-20 is an orthonormal change of basis; Hand-k uses its leading columns.
    z20 = modules["hand-20"](obs)
    w = torch.as_tensor(basis["W_robot"], dtype=torch.float32)
    mu = torch.as_tensor(basis["mu_robot"], dtype=torch.float32)
    check("hand20_reconstruction_max_abs", float((z20 @ w.T + mu - expert_action).abs().max()), 1e-5)
    for k in (2, 4, 8):
        check(f"hand{k}_prefix_of_hand20", float((modules[f"hand-{k}"](obs) - z20[:, :k]).abs().max()), 1e-6)
    sample_var = z20.var(dim=0, unbiased=False).numpy()
    relative = np.abs(sample_var - basis["eigenvalues_robot"]) / np.maximum(basis["eigenvalues_robot"], 1e-3)
    report["hand20_latent_variance_rel_error_top8"] = relative[:8].tolist()
    check("hand20_latent_variance_rel_error_top8_max", float(relative[:8].max()), 0.05)

    # 2b. Joint-K: the same expert action in the joint-position basis, with its own npz keys.
    if args.joint_basis:
        joint_basis = np.load(args.joint_basis)
        wq = torch.as_tensor(joint_basis["W_joint"], dtype=torch.float32)
        muq = torch.as_tensor(joint_basis["mu_joint"], dtype=torch.float32)
        check("joint_basis_orthonormal", float((wq.T @ wq - torch.eye(wq.shape[1])).abs().max()), 1e-5)
        j20 = modules["joint-20"](obs)
        check("joint20_reconstruction_max_abs", float((j20 @ wq.T + muq - expert_action).abs().max()), 1e-5)
        check("joint20_matches_formula", float((j20 - (expert_action - muq) @ wq).abs().max()), 1e-5)
        for k in (2, 4):
            check(f"joint{k}_prefix_of_joint20", float((modules[f"joint-{k}"](obs) - j20[:, :k]).abs().max()), 1e-6)
        wrong_files = 0
        for name, path in (("joint-2", Path(args.pca_basis)), ("hand-2", Path(args.joint_basis))):
            try:
                build_intent(name, **{**common, "pca_basis": path})
            except KeyError:
                wrong_files += 1
        check("wrong_basis_file_rejected_missing", float(2 - wrong_files), 0.0)
        report["joint_basis"] = {"path": str(Path(args.joint_basis).resolve()),
                                 "joint20_latent_std_top4": j20.std(dim=0)[:4].tolist(),
                                 "hand20_latent_std_top4": z20.std(dim=0)[:4].tolist()}

    # 3. Wrist intents.
    check("wrist2_equals_expert_wrist", float((modules["wrist-2"](obs) - expert_action[:, WRIST]).abs().max()), 0.0)
    wf = modules["wrist2-finger-2"](obs)
    wf_expected = (expert_action[:, FINGERS] - torch.as_tensor(basis["mu_finger"], dtype=torch.float32)) @ torch.as_tensor(
        basis["W_finger"][:, :2], dtype=torch.float32
    )
    check("wrist2_finger2_layout", float((wf - torch.cat([expert_action[:, WRIST], wf_expected], 1)).abs().max()), 1e-6)

    object_rot, goal_rot = obs[:, OBJECT_ROT], obs[:, GOAL_ROT]
    if not args.face_turn:
        # 4. Object intents against the environment's own relative-rotation observation.
        theta = modules["obj-1"](obs)[:, 0]
        env_relative = obs[:, 68:72]
        theta_env = 2.0 * torch.asin(torch.clamp(env_relative[:, 1:].norm(dim=1), max=1.0))
        # The quaternion itself matches tightly; the angle is looser because asin amplifies float32
        # rounding as theta approaches pi.
        own_relative = quat_mul(obs[:, OBJECT_ROT], quat_conjugate(obs[:, GOAL_ROT]))
        sign_free = torch.minimum((own_relative - env_relative).abs().amax(1), (own_relative + env_relative).abs().amax(1))
        check("obj_relative_quat_vs_env", float(sign_free.max()), 1e-5)
        check("obj1_vs_env_relative_rotation", float((theta - theta_env).abs().max()), 2e-3)
        report["obj1_theta_quantiles"] = np.quantile(theta.numpy(), [0, 0.1, 0.5, 0.9, 1]).tolist()

        subgoal = modules["obj-4"](obs)
        expected_step = torch.minimum(theta, torch.full_like(theta, args.obj_lookahead))
        step_angle = rotation_distance(subgoal, object_rot)
        remaining_after = rotation_distance(subgoal, goal_rot)
        check("obj4_unit_norm", float((subgoal.norm(dim=1) - 1).abs().max()), 1e-5)
        check("obj4_w_nonnegative", float((-subgoal[:, 0]).clamp(min=0).max()), 0.0)
        check("obj4_step_angle_equals_min_lookahead_theta", float((step_angle - expected_step).abs().max()), 2e-3)
        check("obj4_on_geodesic", float((step_angle + remaining_after - theta).abs().max()), 2e-3)
        rotvec = modules["obj-3"](obs)
        check("obj3_norm_equals_step_angle", float((rotvec.norm(dim=1) - expected_step).abs().max()), 2e-3)
        applied = quat_mul(
            torch.cat([torch.cos(rotvec.norm(dim=1, keepdim=True) / 2),
                       torch.sin(rotvec.norm(dim=1, keepdim=True) / 2) * rotvec / rotvec.norm(dim=1, keepdim=True).clamp(min=1e-8)], 1),
            object_rot,
        )
        check("obj3_rotates_object_to_subgoal", float(rotation_distance(applied, subgoal).max()), 2e-3)

    # 4b. Goal intents.
    goal4 = modules["goal-4"](obs)
    check("goal4_equals_canonical_goal", float((goal4 - canonical_quat(goal_rot)).abs().max()), 0.0)
    check("goal4_w_nonnegative", float((-goal4[:, 0]).clamp(min=0).max()), 0.0)
    check("goal4_unit_norm", float((goal4.norm(dim=1) - 1).abs().max()), 1e-5)
    if args.check_goal_yaw or args.face_turn:
        # Pure yaw goals: q = +-(cos(psi/2), 0, 0, sin(psi/2)); (sin psi, cos psi) must give q back.
        check("goal_quat_xy_abs_max", float(goal_rot[:, 1:3].abs().max()), 1e-3)
        yaw = modules["goal-yaw"](obs)
        check("goal_yaw_unit_norm", float((yaw.norm(dim=1) - 1).abs().max()), 1e-5)
        psi = torch.atan2(yaw[:, 0], yaw[:, 1])
        rebuilt = torch.stack([torch.cos(psi / 2), torch.zeros_like(psi), torch.zeros_like(psi), torch.sin(psi / 2)], 1)
        unit_goal = goal_rot / goal_rot.norm(dim=1, keepdim=True)
        sign_free = torch.minimum((rebuilt - unit_goal).abs().amax(1), (rebuilt + unit_goal).abs().amax(1))
        check("goal_yaw_reconstructs_goal_quat", float(sign_free.max()), 1e-5)
        report["goal_yaw_psi_quantiles"] = np.quantile(psi.numpy(), [0, 0.1, 0.5, 0.9, 1]).tolist()
    if args.face_turn:
        # Hinge angle rebuilt from the goal fields: theta - psi must be the hinge error the reward uses,
        # and the goals must sit on the 90 degree detents.
        hinge = hinge_angle_obs(obs)
        check("hinge_obs_unit_norm", float((hinge.norm(dim=1) - 1).abs().max()), 1e-5)
        theta_h = torch.atan2(hinge[:, 0], hinge[:, 1])
        error = torch.atan2(torch.sin(theta_h - psi), torch.cos(theta_h - psi)).abs()
        error_env = 2.0 * torch.asin(torch.clamp(obs[:, 68:72][:, 1:].norm(dim=1), max=1.0))
        check("hinge_error_vs_env", float((error - error_env).abs().max()), 2e-3)
        check("hinge_error_quat_xy_abs_max", float(obs[:, 69:71].abs().max()), 1e-3)
        detent = torch.round(psi / (np.pi / 2)) * (np.pi / 2)
        check("goal_on_quarter_turn_detent", float((psi - detent).abs().max()), 1e-3)
        report["hinge_error_quantiles"] = np.quantile(error.numpy(), [0, 0.1, 0.5, 0.9, 1]).tolist()
        report["hinge_theta_quantiles"] = np.quantile(theta_h.numpy(), [0, 0.1, 0.5, 0.9, 1]).tolist()

    # 5. Shapes and finiteness for every condition.
    shapes = {}
    for name, module in modules.items():
        out = module(obs)
        shapes[name] = {"dim": module.dim, "shape": list(out.shape), "finite": bool(torch.isfinite(out).all())}
        if out.shape != (obs.shape[0], module.dim) or not shapes[name]["finite"]:
            failures.append(f"shape_{name}")
        if module.dim:
            shapes[name]["abs_max"] = float(out.abs().max())
            shapes[name]["std"] = out.std(dim=0).tolist()
        print(f"intent {name} dim={module.dim} shape={tuple(out.shape)} finite={shapes[name]['finite']}")
    report["intents"] = shapes
    report["module_descriptions"] = {name: module.describe() for name, module in modules.items()}
    report["obj_lookahead_rad"] = args.obj_lookahead
    report["failures"] = failures

    Path(args.output_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"intent_validation_failures={len(failures)}")
    print(f"intent_validation_status={'PASS' if not failures else 'FAIL'}")
    if failures:
        raise SystemExit(1)


def validate_unified(args, dataset_dir, rows, obs, stored_action, basis, common, report, failures, check) -> None:
    """E3 checks on the v4 teacher's rollout: teacher rebuild on 162-D input, hand-K basis, goal-7 against the
    env's own hinge goal and phase kind (recorded by the collector), actor input layout, and rejection of
    intents that are not defined on the unified task."""
    if obs.shape[1] != UNIFIED_OBS_DIM:
        raise ValueError(f"Unified rollout must have {UNIFIED_OBS_DIM}-D observations, got {obs.shape[1]}")
    angle = torch.from_numpy(np.asarray(np.load(dataset_dir / "hinge_angle.npy", mmap_mode="r")[rows], dtype=np.float32))
    goal = torch.from_numpy(np.asarray(np.load(dataset_dir / "hinge_goal.npy", mmap_mode="r")[rows], dtype=np.float32))
    kind = torch.from_numpy(np.asarray(np.load(dataset_dir / "phase_kind.npy", mmap_mode="r")[rows], dtype=np.int64))
    ext = torch.cat([obs, torch.sin(angle)[:, None], torch.cos(angle)[:, None]], dim=1)
    modules = {name: build_intent(name, **common, unified=True) for name in UNIFIED_INTENTS}
    expert = modules["hand-20"].expert
    report["expert_layer_sizes"] = expert.layer_sizes

    expert_action = expert(obs)
    check("expert_vs_recorded_action_max_abs", float((expert_action - stored_action).abs().max()), EXPERT_TOLERANCE)
    z20 = modules["hand-20"](ext)
    w = torch.as_tensor(basis["W_robot"], dtype=torch.float32)
    mu = torch.as_tensor(basis["mu_robot"], dtype=torch.float32)
    check("hand20_reconstruction_max_abs", float((z20 @ w.T + mu - expert_action).abs().max()), 1e-5)
    for k in (2, 4, 8):
        check(f"hand{k}_prefix_of_hand20", float((modules[f"hand-{k}"](ext) - z20[:, :k]).abs().max()), 1e-6)
    sample_var = z20.var(dim=0, unbiased=False).numpy()
    relative = np.abs(sample_var - basis["eigenvalues_robot"]) / np.maximum(basis["eigenvalues_robot"], 1e-3)
    report["hand20_latent_variance_rel_error_top8"] = relative[:8].tolist()
    check("hand20_latent_variance_rel_error_top8_max", float(relative[:8].max()), 0.05)

    g7 = modules["goal-7"](ext)
    check("goal7_quat_equals_canonical_goal", float((g7[:, :4] - canonical_quat(obs[:, GOAL_ROT])).abs().max()), 0.0)
    goal_sc = torch.stack([torch.sin(goal), torch.cos(goal)], dim=1)
    check("goal7_hinge_goal_vs_env", float((g7[:, 4:6] - goal_sc).abs().max()), 2e-3)
    check("goal7_is_turn_vs_env_kind", float((g7[:, 6] - (kind == TURN_KIND).float()).abs().max()), 0.0)
    detent = torch.round(goal / (np.pi / 2)) * (np.pi / 2)
    check("hinge_goal_on_quarter_turn_detent", float((goal - detent).abs().max()), 1e-4)
    report["fraction_rows_in_turn"] = float((kind == TURN_KIND).float().mean())
    report["phase_kind_counts"] = {int(k): int((kind == k).sum()) for k in kind.unique()}

    actor = unified_actor_obs(ext)
    check("actor_obs_dim", float(abs(actor.shape[1] - UNIFIED_ACTOR_DIM)), 0.0)
    check("actor_keeps_hinge_velocity", float((actor[:, 149] - obs[:, 161]).abs().max()), 0.0)
    check("actor_has_hinge_angle", float((actor[:, 150:152] - ext[:, 162:164]).abs().max()), 0.0)
    rejected = 0
    for name in ("goal-4", "obj-4", "wrist-2"):
        try:
            build_intent(name, **common, unified=True)
        except ValueError:
            rejected += 1
    check("non_unified_intents_rejected_missing", float(3 - rejected), 0.0)

    shapes = {}
    for name, module in modules.items():
        out = module(ext)
        shapes[name] = {"dim": module.dim, "shape": list(out.shape), "finite": bool(torch.isfinite(out).all())}
        if out.shape != (ext.shape[0], module.dim) or not shapes[name]["finite"]:
            failures.append(f"shape_{name}")
        if module.dim:
            shapes[name]["std"] = out.std(dim=0).tolist()
        print(f"intent {name} dim={module.dim} shape={tuple(out.shape)} finite={shapes[name]['finite']}")
    report["intents"] = shapes
    report["module_descriptions"] = {name: module.describe() for name, module in modules.items()}
    report["extended_obs_dim"] = UNIFIED_EXT_DIM
    report["failures"] = failures
    Path(args.output_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"intent_validation_failures={len(failures)}")
    print(f"intent_validation_status={'PASS' if not failures else 'FAIL'}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
