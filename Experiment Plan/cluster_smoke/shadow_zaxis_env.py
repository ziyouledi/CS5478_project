"""M2 environment: the Stage 1 Shadow Hand cube task with goals restricted to rotations about world z.

Two changes from InHandManipulationEnv, nothing else:
  * goals are uniform yaw angles in [-pi, pi) about world z (the palm normal), instead of x-then-y rotations;
  * the cube starts flat with a uniform random yaw, instead of a random x-then-y orientation,
    so no goal ever requires flipping the cube.
Reward, success test (full 3-D rotation distance <= success_tolerance), observations and actions are inherited.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.utils.math import quat_from_angle_axis, sample_uniform
from isaaclab_tasks.direct.inhand_manipulation.inhand_manipulation_env import InHandManipulationEnv


class ShadowHandZAxisEnv(InHandManipulationEnv):
    def _random_yaw(self, env_ids) -> torch.Tensor:
        angle = sample_uniform(-math.pi, math.pi, (len(env_ids),), device=self.device)
        return quat_from_angle_axis(angle, self.z_unit_tensor[env_ids])

    def _reset_target_pose(self, env_ids):
        self.goal_rot[env_ids] = self._random_yaw(env_ids)
        goal_pos = self.goal_pos + self.scene.env_origins
        self.goal_markers.visualize(goal_pos, self.goal_rot)

        self.reset_goal_buf[env_ids] = 0

    def _reset_idx(self, env_ids: Sequence[int] | None):
        # Same as InHandManipulationEnv._reset_idx except for the object orientation line.
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        super(InHandManipulationEnv, self)._reset_idx(env_ids)

        self._reset_target_pose(env_ids)

        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        object_default_state[:, 0:3] = (
            object_default_state[:, 0:3] + self.cfg.reset_position_noise * pos_noise + self.scene.env_origins[env_ids]
        )
        object_default_state[:, 3:7] = self._random_yaw(env_ids)

        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        self.object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)

        delta_max = self.hand_dof_upper_limits[env_ids] - self.hand.data.default_joint_pos[env_ids]
        delta_min = self.hand_dof_lower_limits[env_ids] - self.hand.data.default_joint_pos[env_ids]

        dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise
        dof_pos = self.hand.data.default_joint_pos[env_ids] + self.cfg.reset_dof_pos_noise * rand_delta

        dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        dof_vel = self.hand.data.default_joint_vel[env_ids] + self.cfg.reset_dof_vel_noise * dof_vel_noise

        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.hand_dof_targets[env_ids] = dof_pos

        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        self.successes[env_ids] = 0
        self._compute_intermediate_values()
