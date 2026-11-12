"""RL-Games env wrapper that appends an intent vector; import only after AppLauncher has started.

Used by both train_student.py and evaluate_students.py.
"""

from __future__ import annotations

import numpy as np
import torch

from isaaclab_rl.rl_games import RlGamesVecEnvWrapper

from intent_modules import GOAL_FIELDS, GOAL_HIDDEN_OBS_DIM, OBS_DIM, IntentChannel, IntentModule, hide_goal


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

    ``intent_delay`` = d and ``intent_bits`` = b (evaluation only, rate sweep phase 2) delay the intent by
    d control steps within the episode or quantise it to 2**b levels; see ``IntentChannel``. At most one of
    hold, delay and bits may be active.
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
    ):
        self._intent = intent
        self._channel = IntentChannel(hold=intent_hold, delay=intent_delay, bits=intent_bits)
        self._goal_hidden = goal_hidden
        self._asymmetric_critic = asymmetric_critic
        self._checked_shape = False
        super().__init__(env, rl_device, clip_obs, clip_actions)

    @property
    def actor_obs_dim(self) -> int:
        return (GOAL_HIDDEN_OBS_DIM if self._goal_hidden else OBS_DIM) + self._intent.dim

    @property
    def observation_space(self):
        base = super().observation_space
        if base.shape != (OBS_DIM,):
            raise ValueError(f"Expected a {OBS_DIM}-D policy observation, got {base.shape}")
        low, high = base.low, base.high
        if self._goal_hidden:
            goal = np.arange(GOAL_FIELDS.start, GOAL_FIELDS.stop)
            low, high = np.delete(low, goal), np.delete(high, goal)
        low = np.concatenate([low, np.full(self._intent.dim, -np.inf, dtype=np.float32)])
        high = np.concatenate([high, np.full(self._intent.dim, np.inf, dtype=np.float32)])
        # The base wrapper type-checks against legacy gym.spaces.Box, not gymnasium's; keep its class.
        return type(base)(low, high, dtype=np.float32)

    @property
    def state_space(self):
        base = super().state_space
        if not self._asymmetric_critic:
            return base
        # The env has no "critic" group, so the base space is empty; the critic gets the policy observation.
        return type(base)(-self._clip_obs, self._clip_obs, (OBS_DIM,), dtype=np.float32)

    def _process_obs(self, obs_dict):
        processed = super()._process_obs(obs_dict)
        obs = processed["obs"]
        intent = self._intent(obs)
        step = None if self._channel.passthrough else self.unwrapped.episode_length_buf
        intent = self._channel(intent, step)
        actor_obs = hide_goal(obs) if self._goal_hidden else obs
        processed["obs"] = torch.cat([actor_obs, intent], dim=1) if self._intent.dim else actor_obs
        if self._asymmetric_critic:
            processed["states"] = obs
        if not self._checked_shape:
            states_shape = tuple(processed["states"].shape) if self._asymmetric_critic else None
            print(
                f"student_obs_shape={tuple(processed['obs'].shape)} intent_dim={self._intent.dim} "
                f"goal_hidden={self._goal_hidden} intent_hold={self.intent_hold} intent_delay={self.intent_delay} "
                f"intent_bits={self.intent_bits} critic_states_shape={states_shape}"
            )
            if processed["obs"].shape[1] != self.actor_obs_dim or not torch.isfinite(intent).all():
                raise RuntimeError("Student observation shape or intent values are invalid")
            if self._asymmetric_critic and processed["states"].shape[1] != OBS_DIM:
                raise RuntimeError("Critic states shape is invalid")
            self._checked_shape = True
        return processed

    # The channel is keyed on the env's episode step counter, which Isaac Lab has already reset to 0
    # for new episodes when observations are computed.

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
        return self._process_obs(obs_dict)
