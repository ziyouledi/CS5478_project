#!/usr/bin/env python3
"""Record zero-shot / OOD generalization closed-loop VLM + 60 Hz Unified v4 demo episodes (with PD hold):
  1. gen0_ood_lighting_warmdim : Zero-shot warm dim illumination (DomeLight intensity=700, warm orange tint)
  2. gen1_ood_lighting_cool    : Zero-shot cool bright illumination (DomeLight intensity=3200, cool blue tint)
  3. gen2_unseen_prompt        : Zero-shot unseen conversational natural-language prompt phrasing

Uses the exact same PD-hold recording pipeline as `record_vlm_closedloop_demo.py` so every clip can be
composited directly by `make_vlm_closedloop_video.py`.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import re
import sys
import time
import urllib.request
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Unified-HingedCube-Z1-Shadow-Direct-v0")
parser.add_argument("--agent", default="rl_games_cfg_entry_point")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--vllm_url", required=True)
parser.add_argument("--vllm_model", required=True)
parser.add_argument("--max_round_steps", type=int, default=120)
parser.add_argument("--min_think_frames", type=int, default=36)
parser.add_argument("--max_think_frames", type=int, default=90)
parser.add_argument("--post_round_settle", type=int, default=15)
parser.add_argument("--final_hold_frames", type=int, default=60)
parser.add_argument("--warmup_renders", type=int, default=3)
parser.add_argument("--first_warmup_renders", type=int, default=20)
parser.add_argument("--antialiasing", default="FXAA")
parser.add_argument("--crop_size", type=int, default=384)
parser.add_argument("--crop_pad", type=float, default=0.30)
parser.add_argument("--max_tokens", type=int, default=512)
parser.add_argument("--seed", type=int, default=20261122)
parser.add_argument("--only", default=None, help="Comma-separated episode names to record (default: all)")
parser.add_argument("--output_dir", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw
from pxr import Gf, UsdGeom, UsdLux, UsdShade
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.inhand_manipulation.inhand_manipulation_env import rotation_distance
from isaaclab_tasks.utils.hydra import hydra_task_config

import hinged_cube_tasks  # noqa: F401
import z1_tasks  # noqa: F401
from cube_state_graph import (
    COLOUR_PAIR_TO_ORIENT,
    DIST_TABLE,
    FACE_COLOUR,
    ORIENT_QUATS,
    ORIENTATIONS,
    SIDES,
    nearest_orientation,
    rad_to_detent,
    state_to_index,
)
from hinged_cube_env import PHASE_KINDS, QUARTER_TURN, TURN_KIND, _wrap
from z1_render_env import CUBE_LABELS, FACE_LABELS

UP_KIND = PHASE_KINDS.index("up")
ANY_KIND = PHASE_KINDS.index("any")
LABEL_INDEX = {name: i + 1 for i, name in enumerate(FACE_LABELS)}
CUBE_IDS = [LABEL_INDEX[name] for name in CUBE_LABELS]
VLM_CAMS = ("c1", "s040", "s220")
CAM_TITLES = {"c1": "VIEW 1 (FRONT-TOP)", "s040": "VIEW 2 (LEFT-SIDE)", "s220": "VIEW 3 (RIGHT-SIDE)"}
STRIP_TO_DETENT = {FACE_COLOUR[SIDES[(-d) % 4]]: d for d in range(4)}

SYS_VISION_FULL = (
    "You are an embodied manipulation planner controlling a Shadow Hand holding a two-layer hinged cube. "
    "The image is a 3-camera montage (VIEW 1 FRONT-TOP, VIEW 2 LEFT-SIDE, VIEW 3 RIGHT-SIDE) at the same moment. "
    "First report the visible coloured side panels in each view (each side has a large 2/3 panel and a narrow 1/3 strip "
    "along its split edge), then deduce the global cube state (`sky_face`, `front_face`, `on_red_panel_strip`), "
    "and finally output the optimal JSON macro-action to reach the target state."
)

GEN_EPISODES = [
    {
        "name": "gen0_ood_lighting_warmdim",
        "gen_tag": "Zero-Shot OOD Lighting (Warm Dim)",
        "instruction": "Solve the scrambled cube under dim warm lighting: align all sides with white up and red front.",
        "init": ("blue", "yellow", 2),
        "target": ("white", "red", 0),
        "light_intensity": 650.0,
        "light_color": (1.0, 0.70, 0.45),
        "prompt_style": "standard",
    },
    {
        "name": "gen1_ood_lighting_cool",
        "gen_tag": "Zero-Shot OOD Lighting (Cool Bright)",
        "instruction": "Solve the scrambled cube under bright cool lighting: align all sides with white up and red front.",
        "init": ("red", "green", 1),
        "target": ("white", "red", 0),
        "light_intensity": 3400.0,
        "light_color": (0.60, 0.78, 1.0),
        "prompt_style": "standard",
    },
    {
        "name": "gen2_unseen_prompt",
        "gen_tag": "Zero-Shot Unseen Instruction Phrasing",
        "instruction": "The cube's top cap is twisted out of place. Unscramble it so every side matches its strip, white facing up and red in front.",
        "init": ("green", "red", 3),
        "target": ("white", "red", 0),
        "light_intensity": 2000.0,
        "light_color": (0.75, 0.75, 0.75),
        "prompt_style": "conversational",
    },
    {
        "name": "gen3_dark_lowlight",
        "gen_tag": "Zero-Shot Dark Low-Light (10x Dimmer)",
        "instruction": "Solve the scrambled cube in dark low light: align all side colors with white on top and red in front.",
        "init": ("blue", "yellow", 2),
        "target": ("white", "red", 0),
        "light_intensity": 200.0,
        "light_color": (0.75, 0.75, 0.75),
        "prompt_style": "standard",
    },
    {
        "name": "gen4_ood_colour_rev",
        "gen_tag": "Zero-Shot OOD Cube Colours (Green/Blue Swapped)",
        "instruction": "Solve the scrambled cube (green/blue swapped): align all sides with white on top and red in front.",
        "init": ("blue", "yellow", 2),
        "target": ("white", "red", 0),
        "light_intensity": 2000.0,
        "light_color": (0.75, 0.75, 0.75),
        "swap_gb": True,
        "prompt_style": "standard",
    },
    {
        "name": "gen5_casual_spoken",
        "gen_tag": "Zero-Shot Casual Spoken Prompt (No Schema Hint)",
        "instruction": "Hey, the top of this cube is twisted—can you line all the side colors back up so it's solved, with white on top and red facing forward?",
        "init": ("blue", "yellow", 2),
        "target": ("white", "red", 0),
        "light_intensity": 2000.0,
        "light_color": (0.75, 0.75, 0.75),
        "prompt_style": "casual_pure",
    },
    {
        "name": "gen6_novel_palette",
        "gen_tag": "Zero-Shot Novel Colour Palette (Orange / Cyan / Purple / Pink)",
        "instruction": "Solve the novel-palette cube (orange/cyan/purple/pink): align all side colors with white on top and orange in front.",
        "init": ("blue", "yellow", 2),
        "target": ("white", "red", 0),
        "light_intensity": 2000.0,
        "light_color": (0.75, 0.75, 0.75),
        "palette": "novel",
        "prompt_style": "novel_palette",
    },
]


def on_red_strip(detent: int) -> str:
    return FACE_COLOUR[SIDES[(-int(detent)) % 4]]


def state_to_dict(oid: int, det: int) -> dict:
    return {
        "sky_face": ORIENTATIONS[oid]["up_colour"],
        "front_face": ORIENTATIONS[oid]["front_colour"],
        "on_red_panel_strip": on_red_strip(det),
    }


def label_lut(info, device) -> torch.Tensor:
    if isinstance(info, list):
        info = info[0]
    id_to_labels = (info or {}).get("idToLabels", {})
    max_id = max([int(k) for k in id_to_labels] + [0])
    lut = torch.zeros(max_id + 2, dtype=torch.long, device=device)
    for key, value in id_to_labels.items():
        cls = value.get("class", "") if isinstance(value, dict) else str(value)
        names = [c.strip() for c in cls.split(",")]
        hits = [LABEL_INDEX[n] for n in names if n in LABEL_INDEX]
        if hits:
            lut[int(key)] = hits[0]
    return lut


def extract_json(text: str) -> dict | None:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        i, j = text.find("{"), text.rfind("}")
        if i != -1 and j != -1 and j > i:
            text = text[i : j + 1]
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


NOVEL_TO_CANON = {"orange": "red", "cyan": "green", "purple": "yellow", "pink": "blue"}
CANON_TO_NOVEL = {v: k for k, v in NOVEL_TO_CANON.items()}


def _canon_colour(c: str) -> str:
    c = str(c).strip().lower()
    return NOVEL_TO_CANON.get(c, c)


def parse_action(act_obj: dict | None) -> tuple[str, int] | None:
    if not isinstance(act_obj, dict):
        return None
    kind = act_obj.get("action")
    if kind == "turn":
        d = str(act_obj.get("delta", "")).strip()
        if d in ("+90", "90", "+1", "1"):
            return ("turn", 1)
        if d in ("-90", "-1"):
            return ("turn", -1)
        return None
    if kind == "reorient":
        sky = _canon_colour(act_obj.get("sky_face", ""))
        front = _canon_colour(act_obj.get("front_face", ""))
        oid = COLOUR_PAIR_TO_ORIENT.get((sky, front))
        return ("reorient", oid) if oid is not None else None
    return None


def parse_pred_state(state_obj: dict | None) -> tuple[int, int] | None:
    if not isinstance(state_obj, dict):
        return None
    sky = _canon_colour(state_obj.get("sky_face", ""))
    front = _canon_colour(state_obj.get("front_face", ""))
    strip_raw = state_obj.get("on_red_panel_strip", state_obj.get("on_orange_panel_strip", ""))
    strip = _canon_colour(strip_raw)
    oid = COLOUR_PAIR_TO_ORIENT.get((sky, front))
    det = STRIP_TO_DETENT.get(strip)
    return (oid, det) if (oid is not None and det is not None) else None


def query_vllm_one(url: str, model: str, png_bytes: bytes, dst_state: dict,
                   prompt_style: str, instruction: str, max_tokens: int) -> tuple[str, float]:
    b64 = base64.b64encode(png_bytes).decode("ascii")
    if prompt_style == "casual_pure":
        user_txt = instruction
    elif prompt_style == "novel_palette":
        novel_dst = {
            "sky_face": CANON_TO_NOVEL.get(dst_state["sky_face"], dst_state["sky_face"]),
            "front_face": CANON_TO_NOVEL.get(dst_state["front_face"], dst_state["front_face"]),
            "on_orange_panel_strip": CANON_TO_NOVEL.get(dst_state["on_red_panel_strip"], dst_state["on_red_panel_strip"]),
        }
        user_txt = (
            f"This cube uses a novel side-colour palette: orange, cyan, purple, pink (top is white, bottom is magenta).\n"
            f"Target State: {json.dumps(novel_dst)}\n"
            "Observe the 3-camera montage, output `view_obs`, `state` (with `sky_face`, `front_face`, `on_orange_panel_strip`), "
            "`thought`, and the next optimal `action` in JSON."
        )
    elif prompt_style == "conversational":
        user_txt = (
            f"User request: {instruction}\n"
            "(In our state schema, this means sky_face=white, front_face=red, and on_red_panel_strip=red.)\n"
            "Look carefully at the 3 camera views and respond in JSON with `view_obs`, `state`, `thought`, and your next `action`."
        )
    else:
        user_txt = (
            f"Target State: {json.dumps(dst_state)}\n"
            "Observe the 3-camera montage, output `view_obs`, `state`, `thought`, and the next optimal `action` in JSON."
        )
    payload = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYS_VISION_FULL},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": user_txt},
            ]},
        ],
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    dt = time.time() - t0
    return data["choices"][0]["message"]["content"], dt


def build_montage_png(rgb_by_cam: dict[str, np.ndarray], seg_nohand_by_cam: dict[str, np.ndarray],
                      crop_size: int, crop_pad: float) -> bytes:
    views = []
    for cam in VLM_CAMS:
        img = Image.fromarray(rgb_by_cam[cam])
        seg = seg_nohand_by_cam[cam]
        H, W = seg.shape
        ys, xs = np.nonzero(np.isin(seg, CUBE_IDS))
        if len(ys) > 0:
            ymin, ymax = int(ys.min()), int(ys.max())
            xmin, xmax = int(xs.min()), int(xs.max())
            cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
            half = max(max(xmax - xmin + 1, ymax - ymin + 1) * (0.5 + crop_pad), 64.0)
            x0, y0 = max(0, int(round(cx - half))), max(0, int(round(cy - half)))
            x1, y1 = min(W, int(round(cx + half))), min(H, int(round(cy + half)))
        else:
            x0, y0, x1, y1 = W // 4, H // 4, 3 * W // 4, 3 * H // 4
        views.append(img.crop((x0, y0, x1, y1)).resize((crop_size, crop_size), Image.Resampling.BICUBIC))

    header_h, pad, S = 28, 6, crop_size
    canvas = Image.new("RGB", (len(views) * S + (len(views) - 1) * pad, S + header_h), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    for idx, (c, v) in enumerate(zip(VLM_CAMS, views)):
        x_off = idx * (S + pad)
        draw.text((x_off + 10, 7), CAM_TITLES[c], fill=(255, 255, 255))
        canvas.paste(v, (x_off, header_h))
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: dict) -> None:
    out_dir = Path(args_cli.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()

    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 5.0
    env_cfg.scene.clone_in_fabric = False
    env_cfg.seed = args_cli.seed
    env_cfg.sim.render_interval = 10**9
    env_cfg.sim.render.antialiasing_mode = args_cli.antialiasing
    config = agent_cfg["params"]["config"]
    config["num_actors"] = 1
    config["device"] = args_cli.device
    config["device_name"] = args_cli.device
    agent_cfg["params"]["seed"] = args_cli.seed
    clip_obs = float(agent_cfg["params"]["env"].get("clip_observations", math.inf))
    clip_actions = float(agent_cfg["params"]["env"].get("clip_actions", math.inf))

    base_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RlGamesVecEnvWrapper(base_env, args_cli.device, clip_obs, clip_actions)
    vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
    runner = Runner()
    runner.load(agent_cfg)
    player = runner.create_player()
    player.restore(str(checkpoint))
    player.reset()
    player.has_batch_dimension = True

    u = base_env.unwrapped
    u.max_phase_steps = 0
    u.timeout_terminates = False
    u.pose_z1_cameras()
    u.goal_markers.set_visibility(False)
    device = u.device
    stage = u.sim.stage
    hand_prim = UsdGeom.Imageable(stage.GetPrimAtPath("/World/envs/env_0/Robot"))
    light_prim = stage.GetPrimAtPath("/World/Light")
    dome_light = UsdLux.DomeLight(light_prim)
    orient_quats_t = torch.tensor(ORIENT_QUATS, dtype=torch.float32, device=device)
    has_demo_cam = "demo" in u.z1_sensors
    first_render = [True]

    def set_lighting(intensity: float, color: tuple[float, float, float]) -> None:
        dome_light.GetIntensityAttr().Set(float(intensity))
        dome_light.GetColorAttr().Set(Gf.Vec3f(*[float(c) for c in color]))
        for _ in range(5):
            u.sim.render()

    def set_cube_palette(palette: str) -> None:
        """Set cube side-face shader diffuse colors in USD: 'canonical', 'swap_gb', or 'novel' (orange/cyan/purple/pink)."""
        if palette == "novel":
            cols = {
                "_px/": Gf.Vec3f(0.95, 0.45, 0.05),  # orange
                "_py/": Gf.Vec3f(0.05, 0.78, 0.85),  # cyan
                "_nx/": Gf.Vec3f(0.55, 0.15, 0.85),  # purple
                "_ny/": Gf.Vec3f(0.95, 0.35, 0.65),  # pink
            }
        elif palette == "swap_gb":
            cols = {
                "_px/": Gf.Vec3f(0.85, 0.08, 0.08),
                "_py/": Gf.Vec3f(0.10, 0.30, 0.90),
                "_nx/": Gf.Vec3f(0.95, 0.80, 0.05),
                "_ny/": Gf.Vec3f(0.10, 0.65, 0.20),
            }
        else:
            cols = {
                "_px/": Gf.Vec3f(0.85, 0.08, 0.08),
                "_py/": Gf.Vec3f(0.10, 0.65, 0.20),
                "_nx/": Gf.Vec3f(0.95, 0.80, 0.05),
                "_ny/": Gf.Vec3f(0.10, 0.30, 0.90),
            }
        for prim in stage.Traverse():
            p = str(prim.GetPath())
            if "/object/" in p and prim.IsA(UsdShade.Shader):
                attr = prim.GetAttribute("inputs:diffuseColor")
                if attr and attr.IsValid():
                    for suffix, rgb in cols.items():
                        if suffix in p:
                            attr.Set(rgb)
                            break
        for _ in range(5):
            u.sim.render()

    def read_state() -> tuple[int, int]:
        quat = u.object_rot[0].detach().cpu().numpy()
        hinge = float(u.object.data.joint_pos[0, 0].detach().cpu().item())
        oid, _ = nearest_orientation(quat)
        det, _ = rad_to_detent(hinge)
        return (oid, det)

    def current_obs() -> torch.Tensor:
        u._compute_intermediate_values()
        return torch.clamp(u.compute_full_observations(), -clip_obs, clip_obs)

    def set_goal(action: tuple[str, int]) -> None:
        ids_t = torch.tensor([0], dtype=torch.long, device=device)
        cur_hinge = u.object.data.joint_pos[ids_t, 0]
        cur_det_rad = _wrap(torch.round(cur_hinge / QUARTER_TURN) * QUARTER_TURN)
        kind, val = action
        if kind == "reorient":
            u.goal_rot[0] = orient_quats_t[int(val)]
            u.hinge_goal[0] = cur_det_rad[0]
            u.kind[0] = UP_KIND if ORIENTATIONS[int(val)]["is_top_up"] else ANY_KIND
        else:
            oid, _ = nearest_orientation(u.object_rot[0].detach().cpu().numpy())
            u.goal_rot[0] = orient_quats_t[oid]
            u.hinge_goal[0] = _wrap(cur_det_rad[0] + float(val) * QUARTER_TURN)
            u.kind[0] = TURN_KIND
        u.phase_steps[ids_t] = 0
        u.hold_count[ids_t] = 0
        u.max_hold_count[ids_t] = 0
        u.phase_start_hinge[ids_t] = cur_hinge
        u.reset_goal_buf[ids_t] = 0
        u._sync_prev_dists(ids_t)

    @torch.no_grad()
    def step_v4(obs: torch.Tensor) -> tuple[torch.Tensor, bool]:
        s_rot, s_h, s_k = u.goal_rot.clone(), u.hinge_goal.clone(), u.kind.clone()
        act = player.get_action(player.obs_to_torch(obs), is_deterministic=True)
        env.step(act)
        goal_dist = torch.norm(u.object_pos[0] - u.in_hand_pos[0], p=2)
        fallen = bool((goal_dist >= u.cfg.fall_dist).item() or u.reset_terminated[0].bool().item())
        u.goal_rot[:] = s_rot
        u.hinge_goal[:] = s_h
        u.kind[:] = s_k
        u.reset_goal_buf[:] = 0
        return current_obs(), fallen

    def current_hold_action() -> torch.Tensor:
        idx = u.actuated_dof_indices
        lower = u.hand_dof_lower_limits[:, idx]
        upper = u.hand_dof_upper_limits[:, idx]
        q = u.hand.data.joint_pos[:, idx]
        return torch.clamp((2.0 * q - upper - lower) / (upper - lower), -1.0, 1.0)

    @torch.no_grad()
    def step_hold(hold_act: torch.Tensor) -> tuple[torch.Tensor, bool]:
        s_rot, s_h, s_k = u.goal_rot.clone(), u.hinge_goal.clone(), u.kind.clone()
        env.step(hold_act)
        goal_dist = torch.norm(u.object_pos[0] - u.in_hand_pos[0], p=2)
        fallen = bool((goal_dist >= u.cfg.fall_dist).item() or u.reset_terminated[0].bool().item())
        u.goal_rot[:] = s_rot
        u.hinge_goal[:] = s_h
        u.kind[:] = s_k
        u.reset_goal_buf[:] = 0
        return current_obs(), fallen

    def grab_video_frames() -> tuple[np.ndarray, np.ndarray]:
        u.sim.render()
        u.z1_sensors["c1"].update(0.0, force_recompute=True)
        c1_rgb = u.z1_sensors["c1"].data.output["rgb"][0, ..., :3].detach().cpu().numpy()
        if has_demo_cam:
            u.z1_sensors["demo"].update(0.0, force_recompute=True)
            demo_rgb = u.z1_sensors["demo"].data.output["rgb"][0, ..., :3].detach().cpu().numpy()
        else:
            demo_rgb = c1_rgb
        return demo_rgb, c1_rgb

    def render_montage() -> bytes:
        n_warm = args_cli.warmup_renders + (args_cli.first_warmup_renders if first_render[0] else 0)
        first_render[0] = False
        for _ in range(n_warm):
            u.sim.render()
        rgb_np = {}
        for cam in VLM_CAMS:
            sensor = u.z1_sensors[cam]
            sensor.update(0.0, force_recompute=True)
            rgb_np[cam] = sensor.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
        hand_prim.MakeInvisible()
        for _ in range(args_cli.warmup_renders):
            u.sim.render()
        seg_np = {}
        for cam in VLM_CAMS:
            sensor = u.z1_sensors[cam]
            sensor.update(0.0, force_recompute=True)
            lut = label_lut(sensor.data.info.get("semantic_segmentation") if isinstance(sensor.data.info, dict)
                            else sensor.data.info, device)
            raw = sensor.data.output["semantic_segmentation"][0, ..., 0].long().clamp(min=0, max=len(lut) - 1)
            seg_np[cam] = lut[raw].to(torch.uint8).detach().cpu().numpy()
        hand_prim.MakeVisible()
        for _ in range(args_cli.warmup_renders):
            u.sim.render()
        return build_montage_png(rgb_np, seg_np, args_cli.crop_size, args_cli.crop_pad)

    def execute_unrecorded(obs: torch.Tensor, action: tuple[str, int], max_steps: int = 120) -> torch.Tensor:
        set_goal(action)
        obs = current_obs()
        hold_cnt = 0
        for _ in range(max_steps):
            obs, fallen = step_v4(obs)
            if fallen:
                return obs
            od = float(torch.abs(rotation_distance(u.object_rot, u.goal_rot))[0])
            hd = float(torch.abs(u._hinge_error())[0])
            if action[0] == "turn":
                within = (od <= u.turn_orient_tol) and (hd <= u.cfg.success_tolerance)
                hold_cnt = hold_cnt + 1 if within else 0
                if hold_cnt >= u.hold_steps:
                    break
            else:
                if od <= u.cfg.success_tolerance and hd <= u.cfg.success_tolerance:
                    break
        return obs

    def setup_initial_state(init_oid: int, init_det: int) -> torch.Tensor:
        white_up_oid = COLOUR_PAIR_TO_ORIENT[("white", "red")]
        for _ in range(5):
            env.reset()
            obs = current_obs()
            obs = execute_unrecorded(obs, ("reorient", white_up_oid), max_steps=100)
            _, cur_det = read_state()
            diff = (init_det - cur_det) % 4
            if diff == 1:
                obs = execute_unrecorded(obs, ("turn", 1), max_steps=120)
            elif diff == 3:
                obs = execute_unrecorded(obs, ("turn", -1), max_steps=120)
            elif diff == 2:
                obs = execute_unrecorded(obs, ("turn", 1), max_steps=120)
                obs = execute_unrecorded(obs, ("turn", 1), max_steps=120)
            obs = execute_unrecorded(obs, ("reorient", init_oid), max_steps=120)
            for _ in range(20):
                obs, _ = step_v4(obs)
            if read_state() == (init_oid, init_det):
                return obs
        return obs

    all_traces = []
    t_all = time.time()
    only_set = set(args_cli.only.split(",")) if args_cli.only else None

    for spec in GEN_EPISODES:
        if only_set is not None and spec["name"] not in only_set:
            continue
        prefix = spec["name"]
        init_oid = COLOUR_PAIR_TO_ORIENT[(spec["init"][0], spec["init"][1])]
        init_det = spec["init"][2]
        dst_oid = COLOUR_PAIR_TO_ORIENT[(spec["target"][0], spec["target"][1])]
        dst_det = spec["target"][2]
        opt_L = int(DIST_TABLE[state_to_index(init_oid, init_det), state_to_index(dst_oid, dst_det)])

        set_lighting(spec["light_intensity"], spec["light_color"])
        pal = spec.get("palette", "swap_gb" if spec.get("swap_gb") else "canonical")
        set_cube_palette(pal)
        print(f"\n=== Recording {prefix} [{spec['gen_tag']}] (L*={opt_L}) ===", flush=True)

        obs = setup_initial_state(init_oid, init_det)
        actual_init = read_state()

        demo_mp4 = out_dir / f"{prefix}_demo.mp4"
        c1_mp4 = out_dir / f"{prefix}_c1.mp4"
        w_demo = imageio.get_writer(str(demo_mp4), fps=60, codec="libx264", quality=8, macro_block_size=None)
        w_c1 = imageio.get_writer(str(c1_mp4), fps=60, codec="libx264", quality=8, macro_block_size=None)
        frame_idx = 0

        def record_step(o: torch.Tensor) -> tuple[torch.Tensor, bool]:
            nonlocal frame_idx
            o, fell = step_v4(o)
            d_rgb, c_rgb = grab_video_frames()
            w_demo.append_data(d_rgb)
            w_c1.append_data(c_rgb)
            frame_idx += 1
            return o, fell

        def record_hold_steps(o: torch.Tensor, n_steps: int) -> tuple[torch.Tensor, bool]:
            nonlocal frame_idx
            hold_act = current_hold_action()
            fell_flag = False
            for _ in range(n_steps):
                goal_dist = float(torch.norm(u.object_pos[0] - u.in_hand_pos[0], p=2).item())
                if goal_dist > 0.08:
                    o, fell = step_v4(o)
                    hold_act = current_hold_action()
                else:
                    o, fell = step_hold(hold_act)
                d_rgb, c_rgb = grab_video_frames()
                w_demo.append_data(d_rgb)
                w_c1.append_data(c_rgb)
                frame_idx += 1
                if fell:
                    fell_flag = True
                    break
            return o, fell_flag

        for _ in range(args_cli.warmup_renders):
            u.sim.render()
        obs, _ = record_hold_steps(obs, 30)

        dst_dict = state_to_dict(dst_oid, dst_det)
        max_rounds = 2 * opt_L + 2
        rounds_log = []
        success = False
        fell_any = False

        for rnd in range(1, max_rounds + 1):
            cur_st = read_state()
            if cur_st == (dst_oid, dst_det):
                success = True
                break

            frame_start = frame_idx
            png_bytes = render_montage()
            montage_name = f"{prefix}_round{rnd:02d}_montage.png"
            (out_dir / montage_name).write_bytes(png_bytes)

            raw_txt, vlm_dt = query_vllm_one(
                args_cli.vllm_url, args_cli.vllm_model, png_bytes, dst_dict,
                spec["prompt_style"], spec["instruction"], args_cli.max_tokens,
            )
            obj = extract_json(raw_txt)
            pred_st = parse_pred_state(obj.get("state") if isinstance(obj, dict) else None)
            act_field = obj.get("action") if isinstance(obj, dict) else None
            act_dict = act_field if isinstance(act_field, dict) else (obj if isinstance(obj, dict) and "action" in obj else None)
            pred_act = parse_action(act_dict)
            state_ok = (pred_st == cur_st)

            legal = False
            if pred_act is not None:
                if pred_act[0] == "reorient":
                    legal = True
                elif pred_act[0] == "turn" and ORIENTATIONS[cur_st[0]]["is_top_up"]:
                    legal = True

            think_frames = max(args_cli.min_think_frames, min(int(round(vlm_dt * 60)), args_cli.max_think_frames))
            obs, fell_any = record_hold_steps(obs, think_frames)
            frame_think_end = frame_idx
            if fell_any:
                break

            exec_steps = 0
            reached = False
            if legal:
                set_goal(pred_act)
                obs = current_obs()
                hold_cnt = 0
                for _ in range(1, args_cli.max_round_steps + 1):
                    obs, fell = record_step(obs)
                    exec_steps += 1
                    if fell:
                        fell_any = True
                        break
                    od = float(torch.abs(rotation_distance(u.object_rot, u.goal_rot))[0])
                    hd = float(torch.abs(u._hinge_error())[0])
                    if pred_act[0] == "turn":
                        within = (od <= u.turn_orient_tol) and (hd <= u.cfg.success_tolerance)
                        hold_cnt = hold_cnt + 1 if within else 0
                        if hold_cnt >= u.hold_steps:
                            reached = True
                            break
                    else:
                        if od <= u.cfg.success_tolerance and hd <= u.cfg.success_tolerance:
                            reached = True
                            break
                if not fell_any:
                    obs, fell_any = record_hold_steps(obs, args_cli.post_round_settle)

            frame_exec_end = frame_idx
            post_st = read_state()
            rounds_log.append({
                "round": rnd,
                "frame_start": frame_start,
                "frame_think_end": frame_think_end,
                "frame_exec_end": frame_exec_end,
                "vlm_latency_s": round(vlm_dt, 3),
                "think_frames": think_frames,
                "exec_steps": exec_steps,
                "reached": reached,
                "gt_state": state_to_dict(*cur_st),
                "post_state": state_to_dict(*post_st),
                "pred_state": state_to_dict(*pred_st) if pred_st is not None else None,
                "state_ok": state_ok,
                "legal": legal,
                "parsed_action": act_dict,
                "parsed_thought": obj.get("thought") if isinstance(obj, dict) else None,
                "vlm_raw": raw_txt,
                "montage_png": montage_name,
            })
            print(
                f"  round {rnd}: state_ok={state_ok} legal={legal} act={obj.get('action') if isinstance(obj, dict) else None} "
                f"vlm={vlm_dt:.2f}s exec={exec_steps}st post={state_to_dict(*post_st)}",
                flush=True,
            )
            if fell_any:
                break
            if post_st == (dst_oid, dst_det):
                success = True
                break

        if not fell_any:
            obs, _ = record_hold_steps(obs, args_cli.final_hold_frames)

        w_demo.close()
        w_c1.close()
        trace = {
            "name": spec["name"],
            "gen_tag": spec["gen_tag"],
            "prefix": prefix,
            "instruction": spec["instruction"],
            "light_intensity": spec["light_intensity"],
            "light_color": list(spec["light_color"]),
            "prompt_style": spec["prompt_style"],
            "init_state": state_to_dict(*actual_init),
            "target_state": dst_dict,
            "optimal_L": opt_L,
            "success": success,
            "fell": fell_any,
            "rounds_used": len(rounds_log),
            "total_frames": frame_idx,
            "fps": 60,
            "demo_mp4": demo_mp4.name,
            "c1_mp4": c1_mp4.name,
            "rounds": rounds_log,
        }
        (out_dir / f"{prefix}_trace.json").write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
        all_traces.append({k: trace[k] for k in ("prefix", "gen_tag", "optimal_L", "success", "fell", "rounds_used")})

    summary = {"model": args_cli.vllm_model, "seconds": round(time.time() - t_all, 1), "episodes": all_traces}
    (out_dir / "gen_demo_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("vlm_gen_demo_summary=" + json.dumps(summary), flush=True)
    print("record_vlm_gen_demo_status=PASS", flush=True)


if __name__ == "__main__":
    import os, traceback
    try:
        main()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except Exception:
        traceback.print_exc()
        os._exit(1)
