"""Stage 1 Shadow Hand cube task with the DexCube replaced by a two-part hinged cube.

The cube has the outer size and total mass measured from the DexCube (read from the reference
evaluation JSON), but is split into a bottom part (2/3 of the height) and a top layer (1/3) that
turns about the cube's z axis. The hinge is held at 0 by a position drive whose torque is capped:
below the cap the cube behaves as one rigid body, above it the top layer slips.

Reward, goals, observations, actions and hand physics are inherited unchanged. The root link frame
sits at the centre of the whole cube, so object position / rotation mean the same as for the DexCube
(rotation is that of the bottom part).

Settings come from environment variables:
  HINGE_REFERENCE_JSON  evaluation JSON of the stock task holding "object_probe" (required)
  HINGE_ASSET_DIR       directory for the generated URDF / USD (required)
  HINGE_STIFFNESS       drive stiffness, N m / rad      (default 5.0)
  HINGE_DAMPING         drive damping, N m s / rad      (default 0.02)
  HINGE_EFFORT_LIMIT    cap on the drive torque, N m    (default 0.1)
  FACE_TURN_HOLD_STEPS  face-turn task only: control steps the hinge must stay within the success
                        tolerance before the goal counts as reached (default 1 = on first contact)

ShadowHandFaceTurnEnv is the face-turn task on the same cube: the goal is a hinge angle 90 degrees
away from the last one, the drive pulls the top layer to the nearest multiple of 90 degrees
(a detent), and the Stage 1 reward formula is applied to the hinge angle instead of the cube rotation.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab_tasks.direct.inhand_manipulation.inhand_manipulation_env import (
    InHandManipulationEnv,
    rotation_distance,
)


HINGE_JOINT = "hinge"
QUARTER_TURN = math.pi / 2.0
# compute_full_observations layout: 24 joint pos, 24 joint vel, object pos 3, rot 4, lin vel 3, ang vel 3,
# goal pos 3, goal rot 4, then the object-to-goal rotation difference.
ROT_DIFF_SLICE = slice(68, 72)


def _box_inertia(mass: float, x: float, y: float, z: float) -> tuple[float, float, float]:
    return (mass * (y * y + z * z) / 12.0, mass * (x * x + z * z) / 12.0, mass * (x * x + y * y) / 12.0)


def _link_xml(name: str, mass: float, size: tuple[float, float, float], z_offset: float, rgba: str) -> str:
    ixx, iyy, izz = _box_inertia(mass, *size)
    box = f'<box size="{size[0]:.9f} {size[1]:.9f} {size[2]:.9f}"/>'
    origin = f'<origin xyz="0 0 {z_offset:.9f}" rpy="0 0 0"/>'
    return f"""  <link name="{name}">
    <inertial>
      {origin}
      <mass value="{mass:.9f}"/>
      <inertia ixx="{ixx:.12e}" ixy="0" ixz="0" iyy="{iyy:.12e}" iyz="0" izz="{izz:.12e}"/>
    </inertial>
    <visual>
      {origin}
      <geometry>{box}</geometry>
      <material name="{name}_colour"><color rgba="{rgba}"/></material>
    </visual>
    <collision>
      {origin}
      <geometry>{box}</geometry>
    </collision>
  </link>
"""


def write_hinged_cube_urdf(path: Path, size: tuple[float, float, float], mass: float) -> dict:
    """Write the URDF; the base link frame is the centre of the whole cube."""
    sx, sy, sz = size
    base_h, top_h = sz * 2.0 / 3.0, sz / 3.0
    base_mass, top_mass = mass * 2.0 / 3.0, mass / 3.0
    split_z = sz / 2.0 - top_h  # height of the split plane above the cube centre
    urdf = (
        '<?xml version="1.0"?>\n<robot name="hinged_cube">\n'
        + _link_xml("base", base_mass, (sx, sy, base_h), -sz / 2.0 + base_h / 2.0, "0.2 0.4 0.8 1")
        + _link_xml("top", top_mass, (sx, sy, top_h), top_h / 2.0, "0.9 0.6 0.1 1")
        + f"""  <joint name="{HINGE_JOINT}" type="continuous">
    <parent link="base"/>
    <child link="top"/>
    <origin xyz="0 0 {split_z:.9f}" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>
</robot>
"""
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(urdf, encoding="utf-8")
    return {
        "size": [sx, sy, sz],
        "mass_total": mass,
        "mass_base": base_mass,
        "mass_top": top_mass,
        "top_inertia_about_hinge": _box_inertia(top_mass, sx, sy, top_h)[2],
    }


class ShadowHandHingedCubeEnv(InHandManipulationEnv):
    hinge_detent = False  # True: the drive target follows the nearest multiple of 90 degrees

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        reference = Path(os.environ["HINGE_REFERENCE_JSON"]).expanduser()
        asset_dir = Path(os.environ["HINGE_ASSET_DIR"]).expanduser()
        self.hinge_stiffness = float(os.environ.get("HINGE_STIFFNESS", "5.0"))
        self.hinge_damping = float(os.environ.get("HINGE_DAMPING", "0.02"))
        self.hinge_effort_limit = float(os.environ.get("HINGE_EFFORT_LIMIT", "0.1"))

        probe = json.loads(reference.read_text(encoding="utf-8"))["object_probe"]
        bbox_min, bbox_max = probe["bbox_min"], probe["bbox_max"]
        size = tuple(float(hi - lo) for lo, hi in zip(bbox_min, bbox_max))
        centre = [0.5 * (lo + hi) for lo, hi in zip(bbox_min, bbox_max)]
        if max(abs(c) for c in centre) > 1e-3 or min(size) <= 0.0:
            raise ValueError(f"Unexpected DexCube bounds: min={bbox_min} max={bbox_max}")
        mass = float(sum(probe["masses"]))
        self._reference_material = probe.get("material")

        urdf_path = asset_dir / "hinged_cube.urdf"
        self.hinge_geometry = write_hinged_cube_urdf(urdf_path, size, mass)
        # Boxes drawn when rendering (see _setup_scene): link, size, z offset in the link frame, colour.
        sx, sy, sz = size
        self._video_boxes = [
            ("base", (sx, sy, sz * 2.0 / 3.0), -sz / 6.0, (0.2, 0.4, 0.8)),
            ("top", (sx, sy, sz / 3.0), sz / 6.0, (0.9, 0.6, 0.1)),
            # dark stripe from the centre to one edge of the top face, so a quarter turn is visible
            ("top", (sx / 2.0, sy / 8.0, 0.001), sz / 3.0, (0.05, 0.05, 0.05)),
        ]

        source = cfg.object_cfg
        cfg.object_cfg = ArticulationCfg(
            prim_path=source.prim_path,
            spawn=sim_utils.UrdfFileCfg(
                asset_path=str(urdf_path),
                usd_dir=str(asset_dir / "usd"),
                force_usd_conversion=True,
                # diagnostic only (missing visuals in videos): HINGE_DEBUG_INSTANCEABLE=0
                make_instanceable=os.environ.get("HINGE_DEBUG_INSTANCEABLE", "1") != "0",
                fix_base=False,
                root_link_name="base",
                merge_fixed_joints=False,
                self_collision=False,
                joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                    target_type="position",
                    gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                        stiffness=self.hinge_stiffness, damping=self.hinge_damping
                    ),
                ),
                rigid_props=source.spawn.rigid_props,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=source.spawn.rigid_props.solver_position_iteration_count,
                    solver_velocity_iteration_count=source.spawn.rigid_props.solver_velocity_iteration_count,
                    sleep_threshold=source.spawn.rigid_props.sleep_threshold,
                    stabilization_threshold=source.spawn.rigid_props.stabilization_threshold,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=source.init_state.pos, rot=source.init_state.rot, joint_pos={HINGE_JOINT: 0.0}
            ),
            actuators={
                "hinge": ImplicitActuatorCfg(
                    joint_names_expr=[HINGE_JOINT],
                    stiffness=self.hinge_stiffness,
                    damping=self.hinge_damping,
                    effort_limit_sim=self.hinge_effort_limit,
                )
            },
        )
        super().__init__(cfg, render_mode, **kwargs)

        if self.object.num_joints != 1 or self.object.num_bodies != 2:
            raise RuntimeError(
                f"Hinged cube has {self.object.num_joints} joints and {self.object.num_bodies} bodies; expected 1 and 2"
            )
        self._apply_reference_material()

        self._hinge_abs_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        self._hinge_abs_max = torch.zeros((), dtype=torch.float64, device=self.device)
        self._hinge_over = {deg: torch.zeros((), dtype=torch.float64, device=self.device) for deg in (5, 15, 45)}
        self._hinge_samples = 0
        print(
            f"hinged_cube stiffness={self.hinge_stiffness} damping={self.hinge_damping} "
            f"effort_limit={self.hinge_effort_limit} geometry={json.dumps(self.hinge_geometry)}"
        )

    def _apply_reference_material(self) -> None:
        """Give every cube shape the friction / restitution measured on the DexCube."""
        if self._reference_material is None:
            print("hinged_cube_material=default (no reference material in probe)")
            return
        view = self.object.root_physx_view
        materials = view.get_material_properties()
        materials[:] = torch.tensor(self._reference_material, dtype=materials.dtype)
        view.set_material_properties(materials, torch.arange(self.num_envs, device="cpu"))
        print(f"hinged_cube_material={self._reference_material}")

    def _setup_scene(self):
        # Same as InHandManipulationEnv._setup_scene, with the object as an articulation.
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = Articulation(self.cfg.object_cfg)
        if self.render_mode is not None and os.environ.get("HINGE_VIDEO_BOXES", "1") != "0":
            # The visuals written by the URDF converter do not show up in recorded videos (job 1143836),
            # so draw the two parts as plain boxes under their links. No physics schema: rendering only.
            for index, (link, box_size, z_offset, colour) in enumerate(self._video_boxes):
                box_cfg = sim_utils.CuboidCfg(
                    size=box_size, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=colour)
                )
                x_offset = box_size[0] / 2.0 if index == 2 else 0.0
                box_cfg.func(
                    f"/World/envs/env_0/object/{link}/video_box_{index}", box_cfg, translation=(x_offset, 0.0, z_offset)
                )
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.hand
        self.scene.articulations["object"] = self.object
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _apply_action(self) -> None:
        super()._apply_action()
        if self.hinge_detent:
            angle = self.object.data.joint_pos
            self.object.set_joint_position_target(torch.round(angle / QUARTER_TURN) * QUARTER_TURN)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        zeros = torch.zeros((len(env_ids), 1), device=self.device)
        self.object.write_joint_state_to_sim(zeros, zeros, env_ids=env_ids)
        self.object.set_joint_position_target(zeros, env_ids=env_ids)
        super()._reset_idx(env_ids)

    def _compute_intermediate_values(self):
        super()._compute_intermediate_values()
        # The DexCube's centre of mass is its centre. The root link here is the bottom part, whose
        # centre of mass is below the cube centre, so report the velocity of the link frame instead.
        self.object_linvel = self.object.data.root_link_lin_vel_w
        self.object_angvel = self.object.data.root_link_ang_vel_w
        if hasattr(self, "_hinge_abs_sum"):
            angle = self.object.data.joint_pos[:, 0].abs().double()
            self._hinge_abs_sum += angle.sum()
            self._hinge_abs_max = torch.maximum(self._hinge_abs_max, angle.max())
            for deg, counter in self._hinge_over.items():
                counter += (angle > math.radians(deg)).sum()
            self._hinge_samples += angle.numel()

    def handoff_state(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Full simulation state of ``env_ids`` (positions relative to the env origin), as restore_handoff_state reads it."""
        return {
            "hand_joint_pos": self.hand.data.joint_pos[env_ids].clone(),
            "hand_joint_vel": self.hand.data.joint_vel[env_ids].clone(),
            "hand_targets": self.cur_targets[env_ids].clone(),
            "object_pos": (self.object.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]).clone(),
            "object_quat": self.object.data.root_quat_w[env_ids].clone(),
            "object_vel": self.object.data.root_vel_w[env_ids].clone(),
            "hinge_pos": self.object.data.joint_pos[env_ids, 0].clone(),
            "hinge_vel": self.object.data.joint_vel[env_ids, 0].clone(),
        }

    def restore_handoff_state(self, env_ids: torch.Tensor, state: dict[str, torch.Tensor]) -> None:
        """Put ``env_ids`` into stored states (one row per env): hand joints and targets, cube pose, velocity, hinge."""
        targets = state["hand_targets"]
        self.hand.write_joint_state_to_sim(state["hand_joint_pos"], state["hand_joint_vel"], env_ids=env_ids)
        self.hand.set_joint_position_target(targets, env_ids=env_ids)
        self.prev_targets[env_ids] = targets
        self.cur_targets[env_ids] = targets
        self.hand_dof_targets[env_ids] = targets
        pose = torch.cat((state["object_pos"] + self.scene.env_origins[env_ids], state["object_quat"]), dim=-1)
        self.object.write_root_pose_to_sim(pose, env_ids)
        self.object.write_root_velocity_to_sim(state["object_vel"], env_ids)
        hinge = state["hinge_pos"].unsqueeze(-1)
        self.object.write_joint_state_to_sim(hinge, state["hinge_vel"].unsqueeze(-1), env_ids=env_ids)
        self.object.set_joint_position_target(torch.round(hinge / QUARTER_TURN) * QUARTER_TURN, env_ids=env_ids)

    def hinge_summary(self) -> dict:
        n = max(self._hinge_samples, 1)
        summary = {
            "stiffness": self.hinge_stiffness,
            "damping": self.hinge_damping,
            "effort_limit": self.hinge_effort_limit,
            "geometry": self.hinge_geometry,
            "samples": self._hinge_samples,
            "mean_abs_angle_deg": math.degrees(float(self._hinge_abs_sum) / n),
            "max_abs_angle_deg": math.degrees(float(self._hinge_abs_max)),
        }
        for deg, counter in self._hinge_over.items():
            summary[f"fraction_over_{deg}_deg"] = float(counter) / n
        return summary


class ShadowHandFaceTurnEnv(ShadowHandHingedCubeEnv):
    """Turn the top layer by 90 degrees relative to the rest of the cube, again and again.

    Differences from the Stage 1 task on the hinged cube, nothing else:
      * the goal is a hinge angle: the previous goal plus or minus 90 degrees (random direction),
        starting from 0 at reset; ``goal_rot`` holds it as a rotation about z, relative to the bottom part;
      * the rotation distance in the reward and in the success test is the hinge angle error
        (same formula, scales, tolerance and bonus as Stage 1);
      * observation entries [68:72] hold the hinge error as a rotation about z instead of the
        cube-to-goal rotation; all other entries are unchanged (object rotation = bottom part);
      * the cube starts flat with a random yaw, top layer facing away from the palm;
      * the hinge has detents at multiples of 90 degrees;
      * with FACE_TURN_HOLD_STEPS > 1 a goal counts as reached only after the hinge has stayed within
        the success tolerance for that many consecutive control steps (the count is not observed).
    The cube's own orientation is not part of the goal; dropping it still ends the episode.
    """

    hinge_detent = True

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.hinge_goal = torch.zeros(self.num_envs, device=self.device)
        self.hinge_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.hinge_rot[:, 0] = 1.0
        self.hold_steps = int(os.environ.get("FACE_TURN_HOLD_STEPS", "1"))
        if self.hold_steps < 1:
            raise ValueError("FACE_TURN_HOLD_STEPS must be >= 1")
        self.hold_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        print(f"face_turn_hold_steps={self.hold_steps}")
        # Optional start states left by another skill (npz written by evaluate_switch.py --capture):
        # each reset uses one with probability FACE_TURN_START_FRACTION, the flat start otherwise.
        self._start_states = None
        self._start_fraction = float(os.environ.get("FACE_TURN_START_FRACTION", "0.5"))
        start_path = os.environ.get("FACE_TURN_START_STATES")
        if start_path:
            import numpy as np

            data = np.load(Path(start_path).expanduser())
            self._start_states = {key: torch.as_tensor(data[key], dtype=torch.float, device=self.device) for key in data.files}
            self._num_start_states = len(self._start_states["hinge_pos"])
            print(f"face_turn_start_states={start_path} count={self._num_start_states} fraction={self._start_fraction}")
        # evaluation diagnostics
        self._turns_done = 0.0
        self._reward_steps = 0
        self._tilt_sum = 0.0
        self._tilt_over_45 = 0.0
        self._hinge_speed_sum = 0.0

    def _reset_target_pose(self, env_ids):
        sign = torch.randint(0, 2, (len(env_ids),), device=self.device).float() * 2.0 - 1.0
        goal = self.hinge_goal[env_ids] + sign * QUARTER_TURN
        self.hinge_goal[env_ids] = torch.atan2(torch.sin(goal), torch.cos(goal))
        self.goal_rot[env_ids] = quat_from_angle_axis(self.hinge_goal[env_ids], self.z_unit_tensor[env_ids])
        self.reset_goal_buf[env_ids] = 0

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        self.hinge_goal[env_ids] = 0.0
        self.hold_count[env_ids] = 0
        super()._reset_idx(env_ids)

        # Replace the random x-then-y start orientation by a flat cube with a random yaw.
        state = self.object.data.default_root_state.clone()[env_ids]
        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        state[:, 0:3] = state[:, 0:3] + self.cfg.reset_position_noise * pos_noise + self.scene.env_origins[env_ids]
        yaw = sample_uniform(-math.pi, math.pi, (len(env_ids),), device=self.device)
        state[:, 3:7] = quat_from_angle_axis(yaw, self.z_unit_tensor[env_ids])
        self.object.write_root_pose_to_sim(state[:, :7], env_ids)

        if self._start_states is not None:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            ids = env_ids[torch.rand(len(env_ids), device=self.device) < self._start_fraction]
            if len(ids) > 0:
                rows = torch.randint(0, self._num_start_states, (len(ids),), device=self.device)
                picked = {key: value[rows] for key, value in self._start_states.items()}
                self.restore_handoff_state(ids, picked)
                hinge = picked["hinge_pos"]
                self.hinge_goal[ids] = _wrap(torch.round(hinge / QUARTER_TURN) * QUARTER_TURN)
                self._reset_target_pose(ids)  # goal = that detent plus or minus 90 degrees
        self._compute_intermediate_values()

    def _compute_intermediate_values(self):
        super()._compute_intermediate_values()
        if hasattr(self, "hinge_rot"):
            self.hinge_rot = quat_from_angle_axis(self.object.data.joint_pos[:, 0], self.z_unit_tensor)

    def _get_rewards(self) -> torch.Tensor:
        # compute_rewards measures rotation_distance(object_rot, goal_rot); give it the hinge rotation.
        bottom_rot = self.object_rot
        self.object_rot = self.hinge_rot
        successes_before = self.successes.clone()
        try:
            reward = super()._get_rewards() if self.hold_steps == 1 else self._get_rewards_with_hold()
        finally:
            self.object_rot = bottom_rot
        self._turns_done += float((self.successes - successes_before).clamp(min=0).sum())
        self._reward_steps += self.num_envs
        # angle between the hinge axis (cube z) and world z: 0 = top layer facing away from the palm
        cube_z = quat_apply(bottom_rot, self.z_unit_tensor)
        tilt = torch.acos(cube_z[:, 2].clamp(-1.0, 1.0))
        self._tilt_sum += float(tilt.sum())
        self._tilt_over_45 += float((tilt > math.radians(45)).sum())
        self._hinge_speed_sum += float(self.object.data.joint_vel[:, 0].abs().sum())
        return reward

    def _get_rewards_with_hold(self) -> torch.Tensor:
        """compute_rewards of inhand_manipulation_env.py, except that a goal is reached only after
        the rotation distance has stayed within the tolerance for ``hold_steps`` consecutive steps."""
        cfg = self.cfg
        goal_dist = torch.norm(self.object_pos - self.in_hand_pos, p=2, dim=-1)
        rot_dist = rotation_distance(self.object_rot, self.goal_rot)

        dist_rew = goal_dist * cfg.dist_reward_scale
        rot_rew = 1.0 / (torch.abs(rot_dist) + cfg.rot_eps) * cfg.rot_reward_scale
        action_penalty = torch.sum(self.actions**2, dim=-1)
        reward = dist_rew + rot_rew + action_penalty * cfg.action_penalty_scale

        within = torch.abs(rot_dist) <= cfg.success_tolerance
        self.hold_count = torch.where(within, self.hold_count + 1, torch.zeros_like(self.hold_count))
        held = self.hold_count >= self.hold_steps
        goal_resets = torch.where(held, torch.ones_like(self.reset_goal_buf), self.reset_goal_buf)
        self.successes[:] = self.successes + goal_resets
        reward = torch.where(goal_resets == 1, reward + cfg.reach_goal_bonus, reward)
        reward = torch.where(goal_dist >= cfg.fall_dist, reward + cfg.fall_penalty, reward)

        resets = torch.where(goal_dist >= cfg.fall_dist, torch.ones_like(self.reset_buf), self.reset_buf)
        num_resets = torch.sum(resets)
        finished_cons_successes = torch.sum(self.successes * resets.float())
        self.consecutive_successes[:] = torch.where(
            num_resets > 0,
            cfg.av_factor * finished_cons_successes / num_resets
            + (1.0 - cfg.av_factor) * self.consecutive_successes,
            self.consecutive_successes,
        )
        self.reset_goal_buf = goal_resets

        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["consecutive_successes"] = self.consecutive_successes.mean()

        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0:
            self.hold_count[goal_env_ids] = 0
            self._reset_target_pose(goal_env_ids)
        return reward

    def hinge_summary(self) -> dict:
        summary = super().hinge_summary()
        summary["hold_steps"] = self.hold_steps
        n = max(self._reward_steps, 1)
        summary.update(
            {
                "face_turns_per_env_step": self._turns_done / n,
                "face_turns_per_600_steps": 600.0 * self._turns_done / n,
                "mean_axis_tilt_deg": math.degrees(self._tilt_sum / n),
                "fraction_axis_tilt_over_45_deg": self._tilt_over_45 / n,
                "mean_abs_hinge_speed_rad_s": self._hinge_speed_sum / n,
            }
        )
        return summary

    def compute_full_observations(self):
        obs = super().compute_full_observations()
        obs[:, ROT_DIFF_SLICE] = quat_mul(self.hinge_rot, quat_conjugate(self.goal_rot))
        return obs


PHASE_KINDS = ("any", "up", "up2", "turn")
TURN_KIND = PHASE_KINDS.index("turn")


def _wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class ShadowHandSwitchEnv(ShadowHandHingedCubeEnv):
    """Evaluation-only env that alternates the reorientation task and the face-turn task on one cube.

    Every env walks through SWITCH_CYCLE (comma-separated phase kinds, default "up,turn") again and
    again; a phase ends when its goal is reached, then the next one starts from the state left behind.
      any   Stage 1 goal from the stock sampler; reached on first contact within the tolerance
      up    Stage 1 goal = top layer up with a random yaw (rotation about z)
      up2   Stage 1 goal = top layer up with yaw 0 or 180 degrees (inside the stock goal family)
      turn  face-turn goal = nearest detent plus or minus 90 degrees, held FACE_TURN_HOLD_STEPS steps
    Observations are those of the task of the current phase (Stage 1 layout, or the face-turn layout of
    ShadowHandFaceTurnEnv), so the matching expert can be run per env; ``current_kind`` tells which.
    A reset starts the cycle again: flat cube with random yaw if the first phase is a turn, the stock
    random orientation otherwise. The reward is 1 when a phase is completed (not a training reward).
    """

    hinge_detent = True

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        names = [name.strip() for name in os.environ.get("SWITCH_CYCLE", "up,turn").split(",")]
        if not names or any(name not in PHASE_KINDS for name in names):
            raise ValueError(f"SWITCH_CYCLE must list kinds from {PHASE_KINDS}, got {names}")
        self.cycle_names = names
        self.hold_steps = int(os.environ.get("FACE_TURN_HOLD_STEPS", "1"))
        super().__init__(cfg, render_mode, **kwargs)
        self.cycle = torch.tensor([PHASE_KINDS.index(name) for name in names], device=self.device)
        self.phase_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.prev_kind = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.phase_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.phase_start_hinge = torch.zeros(self.num_envs, device=self.device)
        self.hold_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.hinge_goal = torch.zeros(self.num_envs, device=self.device)
        self.hinge_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.hinge_rot[:, 0] = 1.0
        self.recording = False
        self.completions: list[torch.Tensor] = []  # rows: kind, prev kind, steps, tilt deg, hinge drift deg, detent error deg
        self.terminals: list[torch.Tensor] = []  # rows: kind, prev kind, steps in phase, 1 = fall / 0 = time-out
        # kinds whose end states are stored (set by evaluate_switch.py --capture) for FACE_TURN_START_STATES
        self.capture_kinds: set[int] = set()
        self.captured: list[dict[str, torch.Tensor]] = []
        print(f"switch_cycle={','.join(names)} face_turn_hold_steps={self.hold_steps}")

    def current_kind(self) -> torch.Tensor:
        return self.cycle[self.phase_index]

    def _tilt_deg(self) -> torch.Tensor:
        cube_z = quat_apply(self.object_rot, self.z_unit_tensor)
        return torch.rad2deg(torch.acos(cube_z[:, 2].clamp(-1.0, 1.0)))

    def _reset_target_pose(self, env_ids):
        """Start the current phase of ``env_ids``: draw its goal from the state the cube is in now."""
        if not hasattr(self, "cycle"):  # not constructed yet
            return super()._reset_target_pose(env_ids)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        kind = self.cycle[self.phase_index[env_ids]]
        hinge = self.object.data.joint_pos[env_ids, 0]

        ids = env_ids[kind == PHASE_KINDS.index("any")]
        if len(ids) > 0:
            InHandManipulationEnv._reset_target_pose(self, ids)
        ids = env_ids[kind == PHASE_KINDS.index("up")]
        if len(ids) > 0:
            yaw = sample_uniform(-math.pi, math.pi, (len(ids),), device=self.device)
            self.goal_rot[ids] = quat_from_angle_axis(yaw, self.z_unit_tensor[ids])
        ids = env_ids[kind == PHASE_KINDS.index("up2")]
        if len(ids) > 0:
            yaw = torch.randint(0, 2, (len(ids),), device=self.device).float() * math.pi
            self.goal_rot[ids] = quat_from_angle_axis(yaw, self.z_unit_tensor[ids])
        turn = kind == TURN_KIND
        ids = env_ids[turn]
        if len(ids) > 0:
            sign = torch.randint(0, 2, (len(ids),), device=self.device).float() * 2.0 - 1.0
            detent = torch.round(hinge[turn] / QUARTER_TURN) * QUARTER_TURN
            self.hinge_goal[ids] = _wrap(detent + sign * QUARTER_TURN)
            self.goal_rot[ids] = quat_from_angle_axis(self.hinge_goal[ids], self.z_unit_tensor[ids])

        self.phase_steps[env_ids] = 0
        self.hold_count[env_ids] = 0
        self.phase_start_hinge[env_ids] = hinge
        self.reset_goal_buf[env_ids] = 0

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if self.recording and len(env_ids) > 0:
            self.terminals.append(
                torch.stack(
                    (
                        self.current_kind()[env_ids].float(),
                        self.prev_kind[env_ids].float(),
                        self.phase_steps[env_ids].float(),
                        self.reset_terminated[env_ids].float(),
                    ),
                    dim=-1,
                ).cpu()
            )
        self.phase_index[env_ids] = 0
        self.prev_kind[env_ids] = -1
        super()._reset_idx(env_ids)  # zeroes the hinge, stock reset, then _reset_target_pose
        if int(self.cycle[0]) == TURN_KIND:
            # same start as ShadowHandFaceTurnEnv: flat cube, random yaw
            state = self.object.data.default_root_state.clone()[env_ids]
            pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
            state[:, 0:3] = state[:, 0:3] + self.cfg.reset_position_noise * pos_noise + self.scene.env_origins[env_ids]
            yaw = sample_uniform(-math.pi, math.pi, (len(env_ids),), device=self.device)
            state[:, 3:7] = quat_from_angle_axis(yaw, self.z_unit_tensor[env_ids])
            self.object.write_root_pose_to_sim(state[:, :7], env_ids)
            self._compute_intermediate_values()

    def _compute_intermediate_values(self):
        super()._compute_intermediate_values()
        if hasattr(self, "hinge_rot"):
            self.hinge_rot = quat_from_angle_axis(self.object.data.joint_pos[:, 0], self.z_unit_tensor)

    def _get_rewards(self) -> torch.Tensor:
        kind = self.current_kind()
        is_turn = kind == TURN_KIND
        rot_dist = torch.where(
            is_turn,
            rotation_distance(self.hinge_rot, self.goal_rot),
            rotation_distance(self.object_rot, self.goal_rot),
        )
        within = torch.abs(rot_dist) <= self.cfg.success_tolerance
        self.hold_count = torch.where(within, self.hold_count + 1, torch.zeros_like(self.hold_count))
        needed = torch.where(is_turn, self.hold_steps, 1)
        done = self.hold_count >= needed
        self.phase_steps += 1
        self.successes[:] = self.successes + done.float()

        done_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(done_ids) > 0:
            hinge = self.object.data.joint_pos[done_ids, 0]
            if self.recording:
                drift = torch.rad2deg(torch.abs(hinge - self.phase_start_hinge[done_ids]))
                detent_error = torch.rad2deg(torch.abs(hinge - torch.round(hinge / QUARTER_TURN) * QUARTER_TURN))
                self.completions.append(
                    torch.stack(
                        (
                            kind[done_ids].float(),
                            self.prev_kind[done_ids].float(),
                            self.phase_steps[done_ids].float(),
                            self._tilt_deg()[done_ids],
                            drift,
                            detent_error,
                        ),
                        dim=-1,
                    ).cpu()
                )
            if self.capture_kinds:
                keep = torch.zeros_like(done_ids, dtype=torch.bool)
                for captured_kind in self.capture_kinds:
                    keep |= kind[done_ids] == captured_kind
                self.captured.append({key: value.cpu() for key, value in self.handoff_state(done_ids[keep]).items()})
            self.prev_kind[done_ids] = kind[done_ids]
            self.phase_index[done_ids] = (self.phase_index[done_ids] + 1) % len(self.cycle)
            self._reset_target_pose(done_ids)
        return done.float()

    def compute_full_observations(self):
        obs = super().compute_full_observations()
        turn = self.current_kind() == TURN_KIND
        obs[turn, ROT_DIFF_SLICE] = quat_mul(self.hinge_rot, quat_conjugate(self.goal_rot))[turn]
        return obs


from isaaclab.utils import configclass  # noqa: E402
from isaaclab_tasks.direct.shadow_hand.shadow_hand_env_cfg import ShadowHandEnvCfg  # noqa: E402

UNIFIED_OBS_DIM = 162  # 157 Stage 1 + sin/cos(hinge_error) + is_turn + hold_count/hold_steps + scaled hinge_vel


@configclass
class ShadowHandUnifiedEnvCfg(ShadowHandEnvCfg):
    observation_space = UNIFIED_OBS_DIM


class ShadowHandUnifiedEnv(ShadowHandHingedCubeEnv):
    """One task for both skills, in the manner of OpenAI's Rubik's cube hand: a goal is a cube
    orientation plus a hinge angle, goals follow one another without a reset, and one policy sees
    the errors to both.

    Phase kinds (same names as ShadowHandSwitchEnv):
      any   new orientation goal from the stock Stage 1 sampler, hinge goal unchanged
      up    new orientation goal = top layer up with a random yaw, hinge goal unchanged
      turn  hinge goal plus or minus 90 degrees, orientation goal unchanged (only after ``up`` / ``turn``)
    Next goal (unless UNIFIED_CYCLE fixes the order): after ``up`` or ``turn`` a ``turn`` with
    probability 0.5; otherwise ``up`` or ``any`` with probability 0.5 each. A phase still open at
    UNIFIED_MAX_PHASE_STEPS (default 120; 0 disables) terminates the episode when
    UNIFIED_TIMEOUT_TERMINATES=1 (default), or resamples the next goal when 0. With
    UNIFIED_START_STATES set (npz from evaluate_switch.py --capture), UNIFIED_START_FRACTION
    (default 0.5) of resets start directly in a ``turn`` phase from a captured ``up`` handoff state.
    Reached when the cube is within the orientation tolerance and the hinge within 0.1 rad of its goal:
    ``any`` / ``up`` on first contact with the Stage 1 tolerance (0.1 rad); ``turn`` after
    FACE_TURN_HOLD_STEPS consecutive steps (default 10) with orientation tolerance
    UNIFIED_TURN_ORIENT_TOL (default 0.4 rad).
    Reward (v4): potential-based progress on orientation error, hinge error, and per-phase maximum
    hold count, plus a small alive offset, distance/action penalties, completion bonus, and fall penalty.
    Observation: 157 Stage 1 entries + sin/cos(hinge_error) + is_turn + hold_count/hold_steps + scaled hinge_vel
    (set UNIFIED_EXTRA_OBS=0 for the legacy 159-D layout).
    """

    hinge_detent = True

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        cycle = os.environ.get("UNIFIED_CYCLE", "").strip()
        self.cycle_names = [name.strip() for name in cycle.split(",")] if cycle else []
        if any(name not in ("any", "up", "turn") for name in self.cycle_names) or self.cycle_names[:1] == ["turn"]:
            raise ValueError(f"UNIFIED_CYCLE must list any / up / turn and not start with turn, got {self.cycle_names}")
        self.hold_steps = int(os.environ.get("FACE_TURN_HOLD_STEPS", "10"))
        self.turn_orient_tol = float(os.environ.get("UNIFIED_TURN_ORIENT_TOL", "0.4"))
        self.max_phase_steps = int(os.environ.get("UNIFIED_MAX_PHASE_STEPS", "120"))
        self.timeout_terminates = os.environ.get("UNIFIED_TIMEOUT_TERMINATES", "1") != "0"
        self.hinge_prog_scale = float(os.environ.get("UNIFIED_HINGE_PROG_SCALE", "30.0"))
        self.orient_prog_scale = float(os.environ.get("UNIFIED_ORIENT_PROG_SCALE", "15.0"))
        self.hold_prog_scale = float(os.environ.get("UNIFIED_HOLD_PROG_SCALE", "5.0"))
        self.alive_bonus = float(os.environ.get("UNIFIED_ALIVE_BONUS", "0.5"))
        self.unified_fall_penalty = float(os.environ.get("UNIFIED_FALL_PENALTY", "-15.0"))
        self.extra_obs = os.environ.get("UNIFIED_EXTRA_OBS", "1") != "0"
        cfg.observation_space = UNIFIED_OBS_DIM if self.extra_obs else 159
        super().__init__(cfg, render_mode, **kwargs)
        self.cycle = (
            torch.tensor([PHASE_KINDS.index(name) for name in self.cycle_names], device=self.device)
            if self.cycle_names
            else None
        )
        self.kind = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.phase_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.prev_kind = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.phase_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.phase_start_hinge = torch.zeros(self.num_envs, device=self.device)
        self.hold_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.max_hold_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.hinge_goal = torch.zeros(self.num_envs, device=self.device)
        self.prev_orient_dist = torch.zeros(self.num_envs, device=self.device)
        self.prev_hinge_dist = torch.zeros(self.num_envs, device=self.device)
        self._start_states = None
        self._start_fraction = float(os.environ.get("UNIFIED_START_FRACTION", "0.5"))
        start_path = os.environ.get("UNIFIED_START_STATES", "").strip()
        if start_path:
            import numpy as np

            data = np.load(Path(start_path).expanduser())
            self._start_states = {
                key: torch.as_tensor(data[key], dtype=torch.float, device=self.device) for key in data.files
            }
            self._num_start_states = len(self._start_states["hinge_pos"])
            print(f"unified_start_states={start_path} count={self._num_start_states} fraction={self._start_fraction}")
        self.recording = False
        self.completions: list[torch.Tensor] = []  # as in ShadowHandSwitchEnv
        self.terminals: list[torch.Tensor] = []
        print(
            f"unified_cycle={','.join(self.cycle_names) or 'random'} face_turn_hold_steps={self.hold_steps} "
            f"turn_orient_tol={self.turn_orient_tol} max_phase_steps={self.max_phase_steps} "
            f"timeout_terminates={int(self.timeout_terminates)} hinge_prog_scale={self.hinge_prog_scale} "
            f"orient_prog_scale={self.orient_prog_scale} hold_prog_scale={self.hold_prog_scale} "
            f"alive_bonus={self.alive_bonus} fall_penalty={self.unified_fall_penalty} obs_dim={cfg.observation_space}"
        )

    def current_kind(self) -> torch.Tensor:
        return self.kind

    def _tilt_deg(self) -> torch.Tensor:
        cube_z = quat_apply(self.object_rot, self.z_unit_tensor)
        return torch.rad2deg(torch.acos(cube_z[:, 2].clamp(-1.0, 1.0)))

    def _hinge_error(self) -> torch.Tensor:
        return _wrap(self.object.data.joint_pos[:, 0] - self.hinge_goal)

    def _sync_prev_dists(self, env_ids) -> None:
        # object_rot first exists at the end of the first stock _reset_idx; the call after it syncs.
        if not hasattr(self, "prev_orient_dist") or not hasattr(self, "object_rot") or len(env_ids) == 0:
            return
        self.prev_orient_dist[env_ids] = torch.abs(
            rotation_distance(self.object_rot[env_ids], self.goal_rot[env_ids])
        )
        self.prev_hinge_dist[env_ids] = torch.abs(self._hinge_error()[env_ids])

    def _begin_phase(self, env_ids, fresh: bool) -> None:
        """Draw the next goal of ``env_ids``; ``fresh`` = first goal after a reset."""
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        any_kind, up_kind = PHASE_KINDS.index("any"), PHASE_KINDS.index("up")
        hinge = self.object.data.joint_pos[env_ids, 0]
        detent = _wrap(torch.round(hinge / QUARTER_TURN) * QUARTER_TURN)
        self.hinge_goal[env_ids] = detent
        if self.cycle is not None:
            kind = self.cycle[self.phase_index[env_ids]]
        else:
            draws = torch.rand((len(env_ids), 2), device=self.device)
            top_up = (
                torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
                if fresh
                else (self.kind[env_ids] != any_kind)
            )
            kind = torch.where(draws[:, 1] < 0.5, up_kind, any_kind)
            kind = torch.where(top_up & (draws[:, 0] < 0.5), TURN_KIND, kind)
        self.kind[env_ids] = kind

        ids = env_ids[kind == any_kind]
        if len(ids) > 0:
            InHandManipulationEnv._reset_target_pose(self, ids)
        ids = env_ids[kind == up_kind]
        if len(ids) > 0:
            yaw = sample_uniform(-math.pi, math.pi, (len(ids),), device=self.device)
            self.goal_rot[ids] = quat_from_angle_axis(yaw, self.z_unit_tensor[ids])
        turn = kind == TURN_KIND
        ids = env_ids[turn]
        if len(ids) > 0:
            sign = torch.randint(0, 2, (len(ids),), device=self.device).float() * 2.0 - 1.0
            self.hinge_goal[ids] = _wrap(detent[turn] + sign * QUARTER_TURN)

        self.phase_steps[env_ids] = 0
        self.hold_count[env_ids] = 0
        self.max_hold_count[env_ids] = 0
        self.phase_start_hinge[env_ids] = hinge
        self.reset_goal_buf[env_ids] = 0
        self._sync_prev_dists(env_ids)

    def _reset_target_pose(self, env_ids):
        if not hasattr(self, "kind"):  # not constructed yet
            return super()._reset_target_pose(env_ids)
        self._begin_phase(env_ids, fresh=True)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if self.recording and len(env_ids) > 0:
            self.terminals.append(
                torch.stack(
                    (
                        self.kind[env_ids].float(),
                        self.prev_kind[env_ids].float(),
                        self.phase_steps[env_ids].float(),
                        self.reset_terminated[env_ids].float(),
                    ),
                    dim=-1,
                ).cpu()
            )
        self.hinge_goal[env_ids] = 0.0
        self.phase_index[env_ids] = 0
        self.prev_kind[env_ids] = -1
        super()._reset_idx(env_ids)  # zeroes the hinge, stock reset, then _reset_target_pose
        if self._start_states is not None and self.cycle is None and len(env_ids) > 0:
            ids = env_ids[torch.rand(len(env_ids), device=self.device) < self._start_fraction]
            if len(ids) > 0:
                rows = torch.randint(0, self._num_start_states, (len(ids),), device=self.device)
                picked = {key: value[rows] for key, value in self._start_states.items()}
                self.restore_handoff_state(ids, picked)
                quat = picked["object_quat"]
                yaw = 2.0 * torch.atan2(quat[:, 3], quat[:, 0])
                self.goal_rot[ids] = quat_from_angle_axis(yaw, self.z_unit_tensor[ids])
                hinge = picked["hinge_pos"]
                detent = _wrap(torch.round(hinge / QUARTER_TURN) * QUARTER_TURN)
                sign = torch.randint(0, 2, (len(ids),), device=self.device).float() * 2.0 - 1.0
                self.hinge_goal[ids] = _wrap(detent + sign * QUARTER_TURN)
                self.kind[ids] = TURN_KIND
                self.prev_kind[ids] = PHASE_KINDS.index("up")
                self.phase_steps[ids] = 0
                self.hold_count[ids] = 0
                self.max_hold_count[ids] = 0
                self.phase_start_hinge[ids] = hinge
                self.reset_goal_buf[ids] = 0
                self._compute_intermediate_values()
        self._sync_prev_dists(env_ids)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        out_of_reach, time_out = super()._get_dones()
        if self.max_phase_steps > 0 and self.timeout_terminates:
            orient_dist = torch.abs(rotation_distance(self.object_rot, self.goal_rot))
            hinge_dist = torch.abs(self._hinge_error())
            is_turn = self.kind == TURN_KIND
            orient_tol = torch.where(is_turn, self.turn_orient_tol, self.cfg.success_tolerance)
            within = (orient_dist <= orient_tol) & (hinge_dist <= self.cfg.success_tolerance)
            next_hold = torch.where(within, self.hold_count + 1, torch.zeros_like(self.hold_count))
            will_finish = next_hold >= torch.where(is_turn, self.hold_steps, 1)
            phase_timed_out = (~will_finish) & ((self.phase_steps + 1) >= self.max_phase_steps)
            out_of_reach = out_of_reach | phase_timed_out
            time_out = time_out & (~out_of_reach)
        return out_of_reach, time_out

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        goal_dist = torch.norm(self.object_pos - self.in_hand_pos, p=2, dim=-1)
        orient_dist = torch.abs(rotation_distance(self.object_rot, self.goal_rot))
        hinge_dist = torch.abs(self._hinge_error())

        orient_prog = self.prev_orient_dist - orient_dist
        hinge_prog = self.prev_hinge_dist - hinge_dist
        self.prev_orient_dist[:] = orient_dist
        self.prev_hinge_dist[:] = hinge_dist

        is_turn = self.kind == TURN_KIND
        orient_tol = torch.where(is_turn, self.turn_orient_tol, cfg.success_tolerance)
        within = (orient_dist <= orient_tol) & (hinge_dist <= cfg.success_tolerance)
        self.hold_count = torch.where(within, self.hold_count + 1, torch.zeros_like(self.hold_count))
        hold_prog = (self.hold_count - self.max_hold_count).clamp(min=0).float() * is_turn.float()
        self.max_hold_count = torch.maximum(self.max_hold_count, self.hold_count)

        reward = (
            self.alive_bonus
            + goal_dist * cfg.dist_reward_scale
            + self.hinge_prog_scale * hinge_prog
            + self.orient_prog_scale * orient_prog
            + self.hold_prog_scale * hold_prog
            + torch.sum(self.actions**2, dim=-1) * cfg.action_penalty_scale
        )

        done = self.hold_count >= torch.where(is_turn, self.hold_steps, 1)
        self.phase_steps += 1
        self.successes[:] = self.successes + done.float()
        fallen = goal_dist >= cfg.fall_dist
        reward = torch.where(done, reward + cfg.reach_goal_bonus, reward)
        reward = torch.where(fallen, reward + self.unified_fall_penalty, reward)

        resets = torch.where(fallen, torch.ones_like(self.reset_buf), self.reset_buf)
        num_resets = torch.sum(resets)
        finished_cons_successes = torch.sum(self.successes * resets.float())
        self.consecutive_successes[:] = torch.where(
            num_resets > 0,
            cfg.av_factor * finished_cons_successes / num_resets
            + (1.0 - cfg.av_factor) * self.consecutive_successes,
            self.consecutive_successes,
        )
        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["consecutive_successes"] = self.consecutive_successes.mean()

        timed_out = (
            (~done)
            & (~self.reset_buf.bool())
            & (not self.timeout_terminates)
            & (self.max_phase_steps > 0)
            & (self.phase_steps >= self.max_phase_steps)
        )
        done_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(done_ids) > 0 and self.recording:
            hinge = self.object.data.joint_pos[done_ids, 0]
            self.completions.append(
                torch.stack(
                    (
                        self.kind[done_ids].float(),
                        self.prev_kind[done_ids].float(),
                        self.phase_steps[done_ids].float(),
                        self._tilt_deg()[done_ids],
                        torch.rad2deg(torch.abs(hinge - self.phase_start_hinge[done_ids])),
                        torch.rad2deg(torch.abs(hinge - torch.round(hinge / QUARTER_TURN) * QUARTER_TURN)),
                    ),
                    dim=-1,
                ).cpu()
            )
        advance_ids = ((done & (~self.reset_buf.bool())) | timed_out).nonzero(as_tuple=False).squeeze(-1)
        if len(advance_ids) > 0:
            self.prev_kind[advance_ids] = self.kind[advance_ids]
            if self.cycle is not None:
                self.phase_index[advance_ids] = (self.phase_index[advance_ids] + 1) % len(self.cycle)
            self._begin_phase(advance_ids, fresh=False)
        return reward

    def compute_full_observations(self):
        obs = super().compute_full_observations()
        error = self._hinge_error()
        parts = [obs, torch.sin(error).unsqueeze(-1), torch.cos(error).unsqueeze(-1)]
        if self.extra_obs:
            is_turn = (self.kind == TURN_KIND).float().unsqueeze(-1)
            hold_frac = (self.hold_count.float() / float(self.hold_steps)).clamp(0.0, 1.0).unsqueeze(-1)
            hinge_vel = (self.cfg.vel_obs_scale * self.object.data.joint_vel[:, 0]).unsqueeze(-1)
            parts.extend((is_turn, hold_frac, hinge_vel))
        return torch.cat(parts, dim=-1)
