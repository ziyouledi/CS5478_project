"""Inspect the runtime action, joint, control, observation, and goal interface."""

import argparse
import inspect

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Inspect the Shadow Hand task interface")
parser.add_argument("--task", default="Isaac-Repose-Cube-Shadow-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401  Registers the task IDs.
from isaaclab_tasks.utils import parse_env_cfg


def shape_text(value):
    if isinstance(value, torch.Tensor):
        return f"shape={list(value.shape)} dtype={value.dtype} device={value.device}"
    return f"type={type(value).__name__}"


env = None
try:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = 0
    env = gym.make(args.task, cfg=env_cfg)
    core = env.unwrapped
    obs, _ = env.reset()

    print("=== TASK_DIMENSIONS ===", flush=True)
    print(f"task={args.task}", flush=True)
    print(f"action_space={env.action_space}", flush=True)
    print(f"observation_space={env.observation_space}", flush=True)
    print(f"cfg_action_space={core.cfg.action_space}", flush=True)
    print(f"cfg_observation_space={core.cfg.observation_space}", flush=True)
    print(f"physics_dt={core.physics_dt}", flush=True)
    print(f"decimation={core.cfg.decimation}", flush=True)
    print(f"control_dt={core.step_dt}", flush=True)
    print(f"control_hz={1.0 / core.step_dt}", flush=True)
    for cfg_name in ("obs_type", "asymmetric_obs", "act_moving_average", "episode_length_s"):
        print(f"cfg_{cfg_name}={getattr(core.cfg, cfg_name, 'missing')}", flush=True)

    print("=== OBSERVATION_RUNTIME ===", flush=True)
    for key, value in obs.items():
        print(f"observation[{key}]={shape_text(value)}", flush=True)

    print("=== ARTICULATION_CANDIDATES ===", flush=True)
    articulation = None
    for name, value in vars(core).items():
        joint_names = getattr(value, "joint_names", None)
        if joint_names:
            print(f"articulation_attr={name} joint_count={len(joint_names)}", flush=True)
            print(f"all_joint_names={list(joint_names)}", flush=True)
            if articulation is None or "hand" in name.lower():
                articulation = value

    cfg_action_names = getattr(core.cfg, "actuated_joint_names", None)
    print(f"cfg_actuated_joint_names={cfg_action_names}", flush=True)

    index_attr = None
    action_indices = None
    for candidate in ("actuated_dof_indices", "actuated_joint_indices", "_actuated_dof_indices"):
        if hasattr(core, candidate):
            index_attr = candidate
            raw_indices = getattr(core, candidate)
            if isinstance(raw_indices, torch.Tensor):
                action_indices = raw_indices.detach().cpu().tolist()
            else:
                action_indices = list(raw_indices)
            break
    print(f"runtime_action_index_attr={index_attr}", flush=True)
    print(f"runtime_action_joint_indices={action_indices}", flush=True)

    if articulation is None:
        raise RuntimeError("Could not find the Shadow Hand articulation on the environment")
    all_joint_names = list(articulation.joint_names)

    if action_indices is not None and len(action_indices) == core.cfg.action_space:
        action_joint_names = [all_joint_names[index] for index in action_indices]
        mapping_source = index_attr
    elif cfg_action_names is not None:
        action_joint_names = list(cfg_action_names)
        mapping_source = "cfg_actuated_joint_names"
    else:
        raise RuntimeError("Could not determine the action-to-joint mapping")

    print("=== ACTION_TO_JOINT_MAP ===", flush=True)
    print(f"action_mapping_source={mapping_source}", flush=True)
    for action_index, joint_name in enumerate(action_joint_names):
        articulation_index = all_joint_names.index(joint_name)
        is_wrist = "WRJ" in joint_name.upper() or "WRIST" in joint_name.upper()
        print(
            f"ACTION_MAP|action_index={action_index}|joint_name={joint_name}|"
            f"articulation_index={articulation_index}|is_wrist={is_wrist}",
            flush=True,
        )

    if cfg_action_names is not None:
        configured_names = list(cfg_action_names)
        print(f"configured_action_names_match_runtime={configured_names == action_joint_names}", flush=True)

    print("=== GOAL_AND_OBJECT_RUNTIME_FIELDS ===", flush=True)
    for name, value in sorted(vars(core).items()):
        lowered = name.lower()
        if any(token in lowered for token in ("goal", "object_pos", "object_rot", "object_ang", "object_lin")):
            if isinstance(value, torch.Tensor):
                print(f"FIELD|name={name}|{shape_text(value)}", flush=True)

    print("=== OBSERVATION_SOURCE ===", flush=True)
    print(inspect.getsource(type(core)._get_observations), flush=True)
    for method_name in (
        "compute_full_observations",
        "compute_reduced_observations",
        "compute_full_state",
        "_pre_physics_step",
        "_apply_action",
    ):
        method = getattr(type(core), method_name, None)
        if method is not None:
            print(f"=== SOURCE_{method_name} ===", flush=True)
            print(inspect.getsource(method), flush=True)

    print("interface_inspection_status=PASS", flush=True)
finally:
    if env is not None:
        env.close()
    simulation_app.close()
