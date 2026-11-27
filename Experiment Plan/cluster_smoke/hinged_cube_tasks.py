"""Register the hinged-cube variant of the Stage 1 task and the face-turn task (see hinged_cube_env.py).

Only gymnasium is imported here, so scripts can import this module before AppLauncher starts.
"""

import gymnasium as gym


HINGED_CUBE_TASK = "Isaac-Repose-HingedCube-Shadow-Direct-v0"

gym.register(
    id=HINGED_CUBE_TASK,
    entry_point="hinged_cube_env:ShadowHandHingedCubeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.shadow_hand.shadow_hand_env_cfg:ShadowHandEnvCfg",
        "rl_games_cfg_entry_point": "isaaclab_tasks.direct.shadow_hand.agents:rl_games_ppo_cfg.yaml",
    },
)

FACE_TURN_TASK = "Isaac-FaceTurn-HingedCube-Shadow-Direct-v0"

gym.register(
    id=FACE_TURN_TASK,
    entry_point="hinged_cube_env:ShadowHandFaceTurnEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.shadow_hand.shadow_hand_env_cfg:ShadowHandEnvCfg",
        "rl_games_cfg_entry_point": "isaaclab_tasks.direct.shadow_hand.agents:rl_games_ppo_cfg.yaml",
    },
)

SWITCH_TASK = "Isaac-Switch-HingedCube-Shadow-Direct-v0"

gym.register(
    id=SWITCH_TASK,
    entry_point="hinged_cube_env:ShadowHandSwitchEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.shadow_hand.shadow_hand_env_cfg:ShadowHandEnvCfg",
        "rl_games_cfg_entry_point": "isaaclab_tasks.direct.shadow_hand.agents:rl_games_ppo_cfg.yaml",
    },
)

UNIFIED_TASK = "Isaac-Unified-HingedCube-Shadow-Direct-v0"

gym.register(
    id=UNIFIED_TASK,
    entry_point="hinged_cube_env:ShadowHandUnifiedEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "hinged_cube_env:ShadowHandUnifiedEnvCfg",
        "rl_games_cfg_entry_point": "isaaclab_tasks.direct.shadow_hand.agents:rl_games_ppo_cfg.yaml",
    },
)
