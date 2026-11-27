#!/usr/bin/env python3
"""Z1 frame capture: run the frozen unified expert (v4) on the Z1 rendering task and save camera frames.

At control steps capture_start, capture_start + capture_every, ... (< --steps) every env is rendered by
every Z1 camera twice without stepping physics: once as is, once with all hands made invisible. Saved:
  images/s{step}_e{env}_{cam}.png and ..._nohand.png   RGB
  labels/s{step}.npz                                   per camera: label maps (envs, H, W) uint8, index =
                                                       1 + z1_render_env.FACE_LABELS index, 0 = anything else
  frames.jsonl                                         one row per (step, env): cube / hinge / goal state,
                                                       orientation labels, per camera per-label pixel counts
                                                       with and without the hand, framing and neighbour checks
  summary.json                                         cameras, timings and the dry-run checks
Physics runs without rendering between captures (render_interval is set beyond the episode), so frames
are the state after the control step. GOAL_LEVEL_RL_PLAN.md "Z1 的渲染设定".
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Unified-HingedCube-Z1-Shadow-Direct-v0")
parser.add_argument("--agent", default="rl_games_cfg_entry_point")
parser.add_argument("--checkpoint", required=True, help="Frozen unified expert .pth")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--seed", type=int, default=20261118)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--capture_start", type=int, default=60)
parser.add_argument("--capture_every", type=int, default=60)
parser.add_argument("--env_spacing", type=float, default=5.0, help="Visual only: keeps neighbouring envs out of frame")
parser.add_argument("--warmup_renders", type=int, default=3, help="Renders before reading each image set")
parser.add_argument("--first_warmup_renders", type=int, default=20, help="Extra renders before the first capture")
parser.add_argument("--stability_check", action="store_true", help="Render the normal view a second time and diff")
parser.add_argument("--antialiasing", default="FXAA", choices=["Off", "FXAA", "DLSS", "TAA", "DLAA"],
                    help="FXAA: no temporal accumulation, so the hand-hidden render has no ghost of the hand "
                         "(the default DLSS left hand ghosts and shifted label edges in dry run 1163428)")
parser.add_argument("--no_png", action="store_true", help="Skip PNG files (label maps and stats only)")
parser.add_argument("--output_dir", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from pxr import UsdGeom
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from isaaclab.utils.math import matrix_from_quat
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import hinged_cube_tasks  # noqa: F401
import z1_tasks  # noqa: F401
from hinged_cube_env import PHASE_KINDS, QUARTER_TURN
from intent_modules import sha256
from z1_render_env import CUBE_LABELS, FACE_LABELS

try:
    from PIL import Image
except ImportError:  # label maps and stats are still written
    Image = None

FACE_DIRS = ["px", "nx", "py", "ny", "pz", "nz"]
LABEL_INDEX = {name: i + 1 for i, name in enumerate(FACE_LABELS)}
NUM_LABELS = len(FACE_LABELS) + 1
CUBE_IDS = [LABEL_INDEX[name] for name in CUBE_LABELS]
# Visibility and the occlusion check use the face plates only: the dark core shows as thin seams whose pixels
# the annotator sometimes leaves unlabelled in one of the two renders (dry run 1163432, env 2 step 120).
PLATE_IDS = [LABEL_INDEX[name] for name in CUBE_LABELS if not name.startswith("core_")]
HAND_ID = LABEL_INDEX["hand"]


def face_name(axis: int, sign: float) -> str:
    return ("p" if sign > 0 else "n") + "xyz"[axis]


def label_lut(info, device) -> tuple[torch.Tensor, dict]:
    """Map the annotator's semantic ids to LABEL_INDEX (0 for anything else)."""
    if isinstance(info, list):
        info = info[0]
    id_to_labels = (info or {}).get("idToLabels", {})
    max_id = max([int(k) for k in id_to_labels] + [0])
    lut = torch.zeros(max_id + 2, dtype=torch.long, device=device)
    seen = {}
    for key, value in id_to_labels.items():
        cls = value.get("class", "") if isinstance(value, dict) else str(value)
        seen[key] = cls
        names = [c.strip() for c in cls.split(",")]
        hits = [LABEL_INDEX[n] for n in names if n in LABEL_INDEX]
        if hits:
            lut[int(key)] = hits[0]
    return lut, seen


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: dict) -> None:
    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = args_cli.env_spacing
    env_cfg.scene.clone_in_fabric = False  # hands are hidden through USD visibility, so every env needs USD prims
    env_cfg.seed = args_cli.seed
    env_cfg.sim.render_interval = 10**9  # render only at captures
    env_cfg.sim.render.antialiasing_mode = args_cli.antialiasing
    config = agent_cfg["params"]["config"]
    config["num_actors"] = args_cli.num_envs
    config["device"] = args_cli.device
    config["device_name"] = args_cli.device
    agent_cfg["params"]["seed"] = args_cli.seed
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    base_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RlGamesVecEnvWrapper(base_env, args_cli.device, clip_obs, clip_actions)
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
    runner = Runner()
    runner.load(agent_cfg)
    player = runner.create_player()
    player.restore(str(checkpoint))
    player.reset()
    player.has_batch_dimension = True

    u = base_env.unwrapped
    device = u.device
    n = u.num_envs
    camera_poses = u.pose_z1_cameras()
    # The stock goal marker (a lettered DexCube floating beside the hand) would show the goal in the image and
    # block side views (dry run 1163428); the goal is given to the model as text instead.
    u.goal_markers.set_visibility(False)
    cam_names = [c[0] for c in u.z1_cameras]
    image = u.z1_image
    stage = u.sim.stage
    hands = [UsdGeom.Imageable(stage.GetPrimAtPath(f"/World/envs/env_{i}/Robot")) for i in range(n)]
    if not all(h.GetPrim().IsValid() for h in hands):
        raise RuntimeError("Hand prims /World/envs/env_i/Robot not found on the USD stage")
    projections = {name: u.z1_projection(name) for name in cam_names}
    intrinsics_check = {}
    for name in cam_names:
        k = u.z1_sensors[name].data.intrinsic_matrices[0].tolist()
        intrinsics_check[name] = {"sensor_fx": k[0][0], "computed_fx": projections[name][2], "sensor_cx": k[0][2]}
    c1_dir = torch.tensor(
        (math.cos(math.radians(u.z1_cameras[0][1])), math.sin(math.radians(u.z1_cameras[0][1])), 0.0), device=device
    )

    def render(times: int) -> None:
        for _ in range(times):
            u.sim.render()

    def read(name: str):
        sensor = u.z1_sensors[name]
        sensor.update(0.0, force_recompute=True)
        out = sensor.data.output
        lut, seen = label_lut(sensor.data.info.get("semantic_segmentation") if isinstance(sensor.data.info, dict)
                              else sensor.data.info, device)
        raw = out["semantic_segmentation"][..., 0].long().clamp(min=0, max=len(lut) - 1)
        return out["rgb"][..., :3].clone(), lut[raw].to(torch.uint8), seen

    def counts(labels: torch.Tensor) -> torch.Tensor:  # (envs, NUM_LABELS)
        offset = torch.arange(n, device=device).view(n, 1, 1) * NUM_LABELS
        return torch.bincount((labels.long() + offset).flatten(), minlength=n * NUM_LABELS).view(n, NUM_LABELS)

    def project(name: str, points_env: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """points (..., 3) in the camera's env frame -> u, v (pixels), depth."""
        eye, rows, fx = projections[name]
        cam = (points_env - eye) @ rows.T
        z = cam[..., 2]
        return fx * cam[..., 0] / z + image / 2.0, fx * cam[..., 1] / z + image / 2.0, z

    def neighbours_in_frame(name: str) -> list[int]:
        """Per env: how many other envs have a cube centre or a hand body inside this camera's frame."""
        origins = u.scene.env_origins
        pts = torch.cat((u.object.data.root_pos_w.unsqueeze(1), u.hand.data.body_pos_w), dim=1)  # (n, P, 3) world
        result = []
        for i in range(n):
            rel = pts - origins[i]
            px, py, z = project(name, rel)
            inside = (z > 0) & (px >= 0) & (px < image) & (py >= 0) & (py < image)
            inside[i] = False
            result.append(int(inside.any(dim=1).sum()))
        return result

    capture_steps = list(range(args_cli.capture_start, args_cli.steps, args_cli.capture_every))
    frames_path = output_dir / "frames.jsonl"
    frames_file = frames_path.open("w", encoding="utf-8")
    timings = {"physics_s": 0.0, "render_s": 0.0, "readout_s": 0.0, "save_s": 0.0, "captures": 0}
    checks = {"hand_pixels_nohand_max": 0, "hand_pixels_normal_max": 0, "cube_missing_nohand": 0,
              "occlusion_violations": 0, "neighbours_in_frame_max": 0, "stability_mean_abs_diff": [],
              "centroid_offset_px": [], "cube_bbox_fraction": []}
    seen_labels = {}

    obs = env.reset()
    first = True
    with torch.no_grad():
        for step in range(1, args_cli.steps + 1):
            t0 = time.perf_counter()
            actions = player.get_action(player.obs_to_torch(obs["obs"] if isinstance(obs, dict) else obs),
                                        is_deterministic=True)
            obs, _, _, _ = env.step(actions)
            torch.cuda.synchronize()
            timings["physics_s"] += time.perf_counter() - t0
            if step not in capture_steps:
                continue

            t0 = time.perf_counter()
            render(args_cli.warmup_renders + (args_cli.first_warmup_renders if first else 0))
            first = False
            timings["render_s"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            normal = {}
            for name in cam_names:
                rgb, labels, seen = read(name)
                normal[name] = (rgb, labels)
                seen_labels.update(seen)
            timings["readout_s"] += time.perf_counter() - t0
            if args_cli.stability_check:
                render(1)
                for name in cam_names:
                    rgb2, _, _ = read(name)
                    diff = (rgb2.float() - normal[name][0].float()).abs().mean().item()
                    checks["stability_mean_abs_diff"].append(diff)
            t0 = time.perf_counter()
            for h in hands:
                h.MakeInvisible()
            render(args_cli.warmup_renders)
            timings["render_s"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            nohand = {name: read(name)[:2] for name in cam_names}
            timings["readout_s"] += time.perf_counter() - t0
            for h in hands:
                h.MakeVisible()

            t0 = time.perf_counter()
            # State in the env frame (envs are not rotated, so world orientations are env orientations).
            rot = matrix_from_quat(u.object_rot)  # columns = cube axes in the env frame
            up = rot[:, 2, :]  # z component of each cube axis
            up_axis = up.abs().argmax(dim=1)
            up_sign = torch.gather(up, 1, up_axis.unsqueeze(1)).squeeze(1)
            front = c1_dir @ rot  # (n, 3): cube axes projected on the C1 viewing azimuth
            front_axis = front.abs().argmax(dim=1)
            front_sign = torch.gather(front, 1, front_axis.unsqueeze(1)).squeeze(1)
            hinge = u.object.data.joint_pos[:, 0]
            detent = torch.round(hinge / QUARTER_TURN)
            cube_pos_env = u.object_pos  # already relative to the env origin
            stats = {}
            for name in cam_names:
                cn, ch = counts(normal[name][1]), counts(nohand[name][1])
                mask = torch.zeros_like(nohand[name][1], dtype=torch.bool)
                for cid in CUBE_IDS:
                    mask |= nohand[name][1] == cid
                rows_any, cols_any = mask.any(dim=2), mask.any(dim=1)
                idx = torch.arange(image, device=device).float()
                area = mask.sum(dim=(1, 2)).clamp(min=1).float()
                cy = (mask.sum(dim=2).float() * idx).sum(dim=1) / area
                cx = (mask.sum(dim=1).float() * idx).sum(dim=1) / area
                pu, pv, _ = project(name, cube_pos_env)
                neighbours = neighbours_in_frame(name)
                stats[name] = (cn.cpu(), ch.cpu(), rows_any.cpu(), cols_any.cpu(), cx.cpu(), cy.cpu(),
                               pu.cpu(), pv.cpu(), neighbours)
            label_arrays = {}
            for name in cam_names:
                label_arrays[f"{name}_normal"] = normal[name][1].cpu().numpy()
                label_arrays[f"{name}_nohand"] = nohand[name][1].cpu().numpy()
            np.savez_compressed(output_dir / "labels" / f"s{step:04d}.npz", **label_arrays)
            if Image is not None and not args_cli.no_png:
                for name in cam_names:
                    a, b = normal[name][0].cpu().numpy(), nohand[name][0].cpu().numpy()
                    for i in range(n):
                        Image.fromarray(a[i]).save(output_dir / "images" / f"s{step:04d}_e{i:03d}_{name}.png")
                        Image.fromarray(b[i]).save(output_dir / "images" / f"s{step:04d}_e{i:03d}_{name}_nohand.png")
            for i in range(n):
                row = {
                    "step": step,
                    "env": i,
                    "episode_step": int(u.episode_length_buf[i]),
                    "kind": PHASE_KINDS[int(u.kind[i])],
                    "phase_steps": int(u.phase_steps[i]),
                    "cube_pos_env": cube_pos_env[i].tolist(),
                    "cube_quat_wxyz": u.object_rot[i].tolist(),
                    "goal_quat_wxyz": u.goal_rot[i].tolist(),
                    "hinge_rad": float(hinge[i]),
                    "hinge_goal_rad": float(u.hinge_goal[i]),
                    "detent": int(detent[i]) % 4,
                    "detent_residual_deg": math.degrees(float(hinge[i] - detent[i] * QUARTER_TURN)),
                    "up_face": face_name(int(up_axis[i]), float(up_sign[i])),
                    "up_tilt_deg": math.degrees(math.acos(min(1.0, abs(float(up_sign[i]))))),
                    "front_face_c1": face_name(int(front_axis[i]), float(front_sign[i])),
                    "cameras": {},
                }
                for name in cam_names:
                    cn, ch, rows_any, cols_any, cx, cy, pu, pv, neighbours = stats[name]
                    cube_normal = int(sum(cn[i, c] for c in PLATE_IDS))
                    cube_nohand = int(sum(ch[i, c] for c in PLATE_IDS))
                    r, c = rows_any[i].nonzero(), cols_any[i].nonzero()
                    bbox = (max(int(r.max() - r.min() + 1), int(c.max() - c.min() + 1)) / image) if len(r) else 0.0
                    offset = math.hypot(float(cx[i] - pu[i]), float(cy[i] - pv[i])) if cube_nohand else None
                    row["cameras"][name] = {
                        "pixels_normal": {FACE_LABELS[k - 1]: int(cn[i, k]) for k in range(1, NUM_LABELS) if cn[i, k]},
                        "pixels_nohand": {FACE_LABELS[k - 1]: int(ch[i, k]) for k in range(1, NUM_LABELS) if ch[i, k]},
                        "cube_visible_fraction": cube_normal / cube_nohand if cube_nohand else None,
                        "cube_bbox_fraction": bbox,
                        "centroid_offset_px": offset,
                        "neighbours_in_frame": neighbours[i],
                    }
                    checks["hand_pixels_nohand_max"] = max(checks["hand_pixels_nohand_max"], int(ch[i, HAND_ID]))
                    checks["hand_pixels_normal_max"] = max(checks["hand_pixels_normal_max"], int(cn[i, HAND_ID]))
                    checks["cube_missing_nohand"] += int(cube_nohand == 0)
                    checks["occlusion_violations"] += int(any(int(cn[i, k]) > int(ch[i, k]) + 20 for k in PLATE_IDS))
                    checks["neighbours_in_frame_max"] = max(checks["neighbours_in_frame_max"], neighbours[i])
                    if offset is not None:
                        checks["centroid_offset_px"].append(offset)
                    checks["cube_bbox_fraction"].append(bbox)
                frames_file.write(json.dumps(row) + "\n")
            frames_file.flush()
            timings["save_s"] += time.perf_counter() - t0
            timings["captures"] += 1
            print(f"z1_capture step={step} captures={timings['captures']}", flush=True)
    frames_file.close()

    def describe(values):
        if not values:
            return None
        arr = np.asarray(values, dtype=float)
        return {"min": float(arr.min()), "median": float(np.median(arr)), "max": float(arr.max())}

    for key in ("stability_mean_abs_diff", "centroid_offset_px", "cube_bbox_fraction"):
        checks[key] = describe(checks[key])
    passed = (
        checks["hand_pixels_nohand_max"] == 0
        and checks["hand_pixels_normal_max"] > 0
        and checks["cube_missing_nohand"] == 0
        and checks["occlusion_violations"] == 0
    )
    summary = {
        "record_type": "z1_capture",
        "task": args_cli.task,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "seed": args_cli.seed,
        "num_envs": n,
        "steps": args_cli.steps,
        "capture_steps": capture_steps,
        "env_spacing": args_cli.env_spacing,
        "image": image,
        "focal_mm": u.z1_focal,
        "antialiasing": args_cli.antialiasing,
        "target_env": list(u.z1_target),
        "cameras": camera_poses,
        "intrinsics_check": intrinsics_check,
        "face_labels": FACE_LABELS,
        "annotator_labels": seen_labels,
        "cube_size": u.hinge_geometry["size"],
        "timings": timings,
        "checks": checks,
        "checks_passed": passed,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("z1_summary=" + json.dumps({"timings": timings, "checks": checks}))
    env.close()
    print(f"z1_capture_status={'PASS' if passed else 'CHECK_FAILED'}")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        print("z1_capture_status=FAIL", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    simulation_app.close()
