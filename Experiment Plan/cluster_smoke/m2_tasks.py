"""Register the M2 task: single-axis goal reorientation about the palm normal (world z).

Only gymnasium is imported here, so scripts can import this module before AppLauncher starts.
The environment class itself lives in shadow_zaxis_env.py and is imported lazily by gym.make.
Reward, observations, actions, physics and the agent config are those of the Stage 1 task.
"""

import gymnasium as gym


M2_TASK = "Isaac-Repose-Cube-Shadow-ZAxis-Direct-v0"

gym.register(
    id=M2_TASK,
    entry_point="shadow_zaxis_env:ShadowHandZAxisEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.shadow_hand.shadow_hand_env_cfg:ShadowHandEnvCfg",
        "rl_games_cfg_entry_point": "isaaclab_tasks.direct.shadow_hand.agents:rl_games_ppo_cfg.yaml",
    },
)
