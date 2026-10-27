"""RL-Games env wrapper that appends an intent vector; import only after AppLauncher has started.

Same behaviour as ``IntentRlGamesVecEnvWrapper`` in train_student.py. That copy stays in place
while the stage-one array job 1024454 reads train_student.py at runtime; merge them afterwards.
"""

from __future__ import annotations

import numpy as np
import torch

from isaaclab_rl.rl_games import RlGamesVecEnvWrapper

from intent_modules import OBS_DIM, IntentModule


class IntentRlGamesVecEnvWrapper(RlGamesVecEnvWrapper):
    """Appends the intent after the standard observation clipping; intent values are not clipped."""

    def __init__(self, env, rl_device: str, clip_obs: float, clip_actions: float, intent: IntentModule):
        self._intent = intent
        self._checked_shape = False
        super().__init__(env, rl_device, clip_obs, clip_actions)

    @property
    def observation_space(self):
        base = super().observation_space
        if base.shape != (OBS_DIM,):
            raise ValueError(f"Expected a {OBS_DIM}-D policy observation, got {base.shape}")
        low = np.concatenate([base.low, np.full(self._intent.dim, -np.inf, dtype=np.float32)])
        high = np.concatenate([base.high, np.full(self._intent.dim, np.inf, dtype=np.float32)])
        # The base wrapper type-checks against legacy gym.spaces.Box, not gymnasium's; keep its class.
        return type(base)(low, high, dtype=np.float32)

    def _process_obs(self, obs_dict):
        processed = super()._process_obs(obs_dict)
        obs = processed["obs"]
        intent = self._intent(obs)
        processed["obs"] = torch.cat([obs, intent], dim=1) if self._intent.dim else obs
        if not self._checked_shape:
            print(f"student_obs_shape={tuple(processed['obs'].shape)} intent_dim={self._intent.dim}")
            if processed["obs"].shape[1] != OBS_DIM + self._intent.dim or not torch.isfinite(intent).all():
                raise RuntimeError("Student observation shape or intent values are invalid")
            self._checked_shape = True
        return processed

    def reset_with_seed(self, seed: int):
        """Reset every environment with a fixed seed so each checkpoint sees the same initial states."""
        obs_dict, _ = self.env.reset(seed=seed)
        return self._process_obs(obs_dict)
