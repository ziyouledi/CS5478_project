"""RL-Games env wrapper that appends an intent vector; import only after AppLauncher has started.

Used by both train_student.py and evaluate_students.py.
"""

from __future__ import annotations

import numpy as np
import torch

from isaaclab_rl.rl_games import RlGamesVecEnvWrapper

from intent_modules import (
    GOAL_FIELDS, GOAL_HIDDEN_OBS_DIM, HINGE_OBS_DIM, OBS_DIM, UNIFIED_ACTOR_DIM, UNIFIED_HINGE_ERROR, UNIFIED_HINGE_VEL,
    UNIFIED_IS_TURN, UNIFIED_OBS_DIM, IntentChannel, IntentModule, hide_goal, hinge_angle_obs, unified_actor_obs,
    unified_hinge_goal,
)

UNIFIED_CHECK_CALLS = 50  # observation calls on which the unified goal reconstruction is checked against the env


class IntentRlGamesVecEnvWrapper(RlGamesVecEnvWrapper):
    """Appends the intent after the standard observation clipping; intent values are not clipped.

    The base wrapper reads the observation space from the unwrapped env, so a gym
    ObservationWrapper cannot change the network input size; this subclass does it here.

    The intent is always computed from the full clipped 157-D observation. With ``goal_hidden`` the
    actor sees that observation without the goal fields (149-D) plus the intent. With
    ``asymmetric_critic`` the wrapper also returns the full clipped 157-D observation as RL-Games
    ``states`` (critic input, no intent); the agent config then needs a ``central_value_config``.

    With ``intent_hold`` = N > 1 (zero-order hold), an environment's intent is recomputed only when
    its episode step count is a multiple of N (always at episode start) and is held otherwise. A goal
    change inside an episode does not trigger a refresh. N = 1 recomputes every step (M1 behaviour).

    ``intent_delay`` = d (evaluation, and training since phase 3) and ``intent_bits`` = b (evaluation only) delay the intent by
    d control steps within the episode or quantise it to 2**b levels; see ``IntentChannel``. At most one of
    hold, delay and bits may be active.

    ``action_hold`` = N > 1 (slow controller, rate sweep phase 4) applies a new policy action only when the
    episode step of the observation it answers is a multiple of N (always at episode start); on the other
    steps the previously applied action is repeated. The policy is still queried every step.

    ``hinge_angle`` (face-turn task only) inserts (sin, cos) of the current hinge angle between the
    observation and the intent. The face-turn observation shows the hinge only through the goal fields, so a
    goal-hidden actor would otherwise not see the state of the top layer. It is computed every step from the
    student's own observation and does not pass through the intent channel.

    ``unified`` (ShadowHandUnifiedEnv, FACE_TURN_PLAN.md E3): the env observation is 162-D. The wrapper appends
    (sin, cos) of the current hinge angle, read from the env, giving the 164-D extended observation the intent
    reads. The actor sees ``unified_actor_obs`` (152-D: goal fields, hinge error, is_turn and hold count removed,
    hinge velocity kept, hinge angle added) plus the intent; the critic sees the clipped 162-D observation.
    Requires goal_hidden, asymmetric_critic and hinge_angle. On the first calls the hinge goal rebuilt from the
    observation and the is_turn bit are checked against the env's own values.
    """

    def __init__(
        self,
        env,
        rl_device: str,
        clip_obs: float,
        clip_actions: float,
        intent: IntentModule,
        goal_hidden: bool = False,
        asymmetric_critic: bool = False,
        intent_hold: int = 1,
        intent_delay: int = 0,
        intent_bits: int = 0,
        action_hold: int = 1,
        hinge_angle: bool = False,
        unified: bool = False,
    ):
        if unified and not (goal_hidden and asymmetric_critic and hinge_angle):
            raise ValueError("unified students need goal_hidden, asymmetric_critic and hinge_angle")
        if unified != getattr(intent, "unified", False):
            raise ValueError(f"Intent module unified={getattr(intent, 'unified', False)} but wrapper unified={unified}")
        self._unified = bool(unified)
        self._task_obs_dim = UNIFIED_OBS_DIM if unified else OBS_DIM
        self._unified_checks = 0
        self.unified_check_max_error = 0.0
        self._intent = intent
        self._hinge_dim = HINGE_OBS_DIM if hinge_angle else 0
        self._channel = IntentChannel(hold=intent_hold, delay=intent_delay, bits=intent_bits)
        self._goal_hidden = goal_hidden
        self._asymmetric_critic = asymmetric_critic
        self._checked_shape = False
        if action_hold < 1:
            raise ValueError(f"Invalid action_hold={action_hold}")
        self._action_hold = int(action_hold)
        self._held_action = None
        super().__init__(env, rl_device, clip_obs, clip_actions)

    @property
    def actor_obs_dim(self) -> int:
        if self._unified:
            return UNIFIED_ACTOR_DIM + self._intent.dim
        return (GOAL_HIDDEN_OBS_DIM if self._goal_hidden else OBS_DIM) + self._hinge_dim + self._intent.dim

    @property
    def observation_space(self):
        base = super().observation_space
        if base.shape != (self._task_obs_dim,):
            raise ValueError(f"Expected a {self._task_obs_dim}-D policy observation, got {base.shape}")
        low, high = base.low, base.high
        if self._unified:
            # same order as unified_actor_obs: [0:64], [72:157], [161], then hinge angle (2) and intent
            keep = np.r_[0:GOAL_FIELDS.start, GOAL_FIELDS.stop:UNIFIED_HINGE_ERROR.start, UNIFIED_HINGE_VEL.start]
            extra = HINGE_OBS_DIM + self._intent.dim
            low = np.concatenate([low[keep], np.full(extra, -np.inf, dtype=np.float32)])
            high = np.concatenate([high[keep], np.full(extra, np.inf, dtype=np.float32)])
            return type(base)(low, high, dtype=np.float32)
        if self._goal_hidden:
            goal = np.arange(GOAL_FIELDS.start, GOAL_FIELDS.stop)
            low, high = np.delete(low, goal), np.delete(high, goal)
        extra = self._hinge_dim + self._intent.dim
        low = np.concatenate([low, np.full(extra, -np.inf, dtype=np.float32)])
        high = np.concatenate([high, np.full(extra, np.inf, dtype=np.float32)])
        # The base wrapper type-checks against legacy gym.spaces.Box, not gymnasium's; keep its class.
        return type(base)(low, high, dtype=np.float32)

    @property
    def state_space(self):
        base = super().state_space
        if not self._asymmetric_critic:
            return base
        # The env has no "critic" group, so the base space is empty; the critic gets the policy observation.
        return type(base)(-self._clip_obs, self._clip_obs, (self._task_obs_dim,), dtype=np.float32)

    def _process_obs(self, obs_dict):
        processed = super()._process_obs(obs_dict)
        obs = processed["obs"]
        if self._unified:
            angle = self.unwrapped.object.data.joint_pos[:, 0].to(obs.device, obs.dtype)
            ext = torch.cat([obs, torch.sin(angle).unsqueeze(1), torch.cos(angle).unsqueeze(1)], dim=1)
            if self._unified_checks < UNIFIED_CHECK_CALLS:
                self._check_unified(ext)
            intent = self._intent(ext)
        else:
            intent = self._intent(obs)
        step = None if self._channel.passthrough else self.unwrapped.episode_length_buf
        intent = self._channel(intent, step)
        if self._unified:
            actor_obs = unified_actor_obs(ext)
        else:
            actor_obs = hide_goal(obs) if self._goal_hidden else obs
            if self._hinge_dim:
                actor_obs = torch.cat([actor_obs, hinge_angle_obs(obs)], dim=1)
        processed["obs"] = torch.cat([actor_obs, intent], dim=1) if self._intent.dim else actor_obs
        if self._asymmetric_critic:
            processed["states"] = obs
        if not self._checked_shape:
            states_shape = tuple(processed["states"].shape) if self._asymmetric_critic else None
            print(
                f"student_obs_shape={tuple(processed['obs'].shape)} intent_dim={self._intent.dim} "
                f"goal_hidden={self._goal_hidden} intent_hold={self.intent_hold} intent_delay={self.intent_delay} "
                f"intent_bits={self.intent_bits} action_hold={self._action_hold} hinge_dim={self._hinge_dim} "
                f"critic_states_shape={states_shape}"
            )
            if processed["obs"].shape[1] != self.actor_obs_dim or not torch.isfinite(intent).all():
                raise RuntimeError("Student observation shape or intent values are invalid")
            if self._asymmetric_critic and processed["states"].shape[1] != self._task_obs_dim:
                raise RuntimeError("Critic states shape is invalid")
            self._checked_shape = True
        return processed

    def _check_unified(self, ext: torch.Tensor) -> None:
        """Hinge goal rebuilt from the observation vs the env's hinge_goal; is_turn vs the env's phase kind."""
        env = self.unwrapped
        goal = env.hinge_goal.to(ext.device, ext.dtype)
        rebuilt = unified_hinge_goal(ext)
        goal_error = (rebuilt - torch.stack([torch.sin(goal), torch.cos(goal)], dim=1)).abs().max()
        from hinged_cube_env import TURN_KIND  # Isaac Sim is running whenever this wrapper is used

        is_turn = (env.kind == TURN_KIND).to(ext.device, ext.dtype)
        turn_error = (ext[:, UNIFIED_IS_TURN.start] - is_turn).abs().max()
        error = float(torch.maximum(goal_error, turn_error))
        self.unified_check_max_error = max(self.unified_check_max_error, error)
        self._unified_checks += 1
        if error > 2e-3:
            raise RuntimeError(f"Unified goal reconstruction differs from the env: hinge goal {float(goal_error):.3e}, "
                               f"is_turn {float(turn_error):.3e}")
        if self._unified_checks == UNIFIED_CHECK_CALLS:
            print(f"unified_goal_check calls={self._unified_checks} max_error={self.unified_check_max_error:.3e} PASS")

    # The channel is keyed on the env's episode step counter, which Isaac Lab has already reset to 0
    # for new episodes when observations are computed.

    @property
    def action_hold(self) -> int:
        return self._action_hold

    def step(self, actions):
        if self._action_hold > 1:
            # episode_length_buf is the episode step of the observation these actions answer.
            step = self.unwrapped.episode_length_buf.to(actions.device).long()
            refresh = (step % self._action_hold) == 0
            if self._held_action is None or self._held_action.shape != actions.shape:
                refresh = torch.ones_like(refresh)
                self._held_action = actions.detach().clone()
            self._held_action = torch.where(refresh.unsqueeze(1), actions.detach(), self._held_action)
            actions = self._held_action
        return super().step(actions)

    @property
    def intent_hold(self) -> int:
        return self._channel.hold

    @property
    def intent_delay(self) -> int:
        return self._channel.delay

    @property
    def intent_bits(self) -> int:
        return self._channel.bits

    def reset_with_seed(self, seed: int):
        """Reset every environment with a fixed seed so each checkpoint sees the same initial states."""
        obs_dict, _ = self.env.reset(seed=seed)
        self._channel.reset()
        self._held_action = None
        return self._process_obs(obs_dict)
