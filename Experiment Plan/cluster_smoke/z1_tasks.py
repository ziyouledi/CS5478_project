"""Register the Z1 rendering variant of the unified hinged-cube task (z1_render_env.py). Imports only gymnasium."""

import gymnasium as gym


Z1_TASK = "Isaac-Unified-HingedCube-Z1-Shadow-Direct-v0"

gym.register(
    id=Z1_TASK,
    entry_point="z1_render_env:ShadowHandUnifiedZ1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "hinged_cube_env:ShadowHandUnifiedEnvCfg",
        "rl_games_cfg_entry_point": "isaaclab_tasks.direct.shadow_hand.agents:rl_games_ppo_cfg.yaml",
    },
)
