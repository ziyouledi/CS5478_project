#!/usr/bin/env python3
"""Real 60 Hz Isaac Lab Unified v4 + online 3-camera rendering + localhost vLLM HTTP closed-loop evaluation.

Runs frozen Unified v4 (`expert-unified-v4-seed0.pth`) inside `Isaac-Unified-HingedCube-Z1-Shadow-Direct-v0`,
renders live 3-view cropped `montage` images (`c1 + s040 + s220`) after each macro-action, queries a local
vLLM OpenAI-compatible server for the next JSON macro-action, and executes it physically at 60 Hz!
"""

from __future__ import annotations

import argparse
import base64
import collections
import concurrent.futures
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
parser.add_argument("--checkpoint", required=True, help="Frozen Unified v4 checkpoint .pth")
parser.add_argument("--vllm_url", required=True, help="e.g. http://127.0.0.1:8000/v1/chat/completions")
parser.add_argument("--vllm_model", required=True, help="Model name served by vLLM")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--batches_per_level", type=int, default=2, help="Batches of num_envs per L (2x16=32 eps/L)")
parser.add_argument("--levels", default="1,2,3,4")
parser.add_argument("--max_round_steps", type=int, default=120)
parser.add_argument("--settle_steps", type=int, default=60)
parser.add_argument("--env_spacing", type=float, default=5.0)
parser.add_argument("--warmup_renders", type=int, default=3)
parser.add_argument("--first_warmup_renders", type=int, default=20)
parser.add_argument("--antialiasing", default="FXAA")
parser.add_argument("--crop_size", type=int, default=384)
parser.add_argument("--crop_pad", type=float, default=0.30)
parser.add_argument("--max_tokens", type=int, default=512)
parser.add_argument("--seed", type=int, default=20261121)
parser.add_argument("--light_intensity", type=float, default=None, help="Override /World/Light DomeLight intensity")
parser.add_argument("--light_color", type=float, nargs=3, default=None, help="Override /World/Light DomeLight RGB color")
parser.add_argument("--prompt_style", choices=["standard", "casual"], default="standard",
                    help="standard: Target State JSON; casual: pure spoken natural-language prompt with zero schema hints")
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
from PIL import Image, ImageDraw
from pxr import Gf, UsdGeom, UsdLux
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
    index_to_state,
    nearest_orientation,
    rad_to_detent,
    state_to_index,
    detent_to_rad,
)
from hinged_cube_env import PHASE_KINDS, QUARTER_TURN, TURN_KIND, _wrap
from z1_render_env import CUBE_LABELS, FACE_LABELS

UP_KIND = PHASE_KINDS.index("up")
ANY_KIND = PHASE_KINDS.index("any")
LABEL_INDEX = {name: i + 1 for i, name in enumerate(FACE_LABELS)}
CUBE_IDS = [LABEL_INDEX[name] for name in CUBE_LABELS]
CAMS = ("c1", "s040", "s220")
CAM_TITLES = {"c1": "VIEW 1 (FRONT-TOP)", "s040": "VIEW 2 (LEFT-SIDE)", "s220": "VIEW 3 (RIGHT-SIDE)"}
STRIP_TO_DETENT = {FACE_COLOUR[SIDES[(-d) % 4]]: d for d in range(4)}

SYS_VISION_FULL = (
    "You are an embodied manipulation planner controlling a Shadow Hand holding a two-layer hinged cube. "
    "The image is a 3-camera montage (VIEW 1 FRONT-TOP, VIEW 2 LEFT-SIDE, VIEW 3 RIGHT-SIDE) at the same moment. "
    "First report the visible coloured side panels in each view (each side has a large 2/3 panel and a narrow 1/3 strip "
    "along its split edge), then deduce the global cube state (`sky_face`, `front_face`, `on_red_panel_strip`), "
    "and finally output the optimal JSON macro-action to reach the target state."
)


def on_red_strip(detent: int) -> str:
    return FACE_COLOUR[SIDES[(-int(detent)) % 4]]


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
        sky = str(act_obj.get("sky_face", "")).strip().lower()
        front = str(act_obj.get("front_face", "")).strip().lower()
        oid = COLOUR_PAIR_TO_ORIENT.get((sky, front))
        return ("reorient", oid) if oid is not None else None
    return None


def parse_pred_state(state_obj: dict | None) -> tuple[int, int] | None:
    if not isinstance(state_obj, dict):
        return None
    sky = str(state_obj.get("sky_face", "")).strip().lower()
    front = str(state_obj.get("front_face", "")).strip().lower()
    strip = str(state_obj.get("on_red_panel_strip", "")).strip().lower()
    oid = COLOUR_PAIR_TO_ORIENT.get((sky, front))
    det = STRIP_TO_DETENT.get(strip)
    return (oid, det) if (oid is not None and det is not None) else None


def query_vllm_one(url: str, model: str, png_bytes: bytes, dst_state: dict, max_tokens: int,
                   prompt_style: str = "standard") -> str:
    b64 = base64.b64encode(png_bytes).decode("ascii")
    if prompt_style == "casual":
        sky, front, strip = dst_state["sky_face"], dst_state["front_face"], dst_state["on_red_panel_strip"]
        if strip == "red":
            user_txt = (
                f"Hey, can you line all the side colors back up so the cube is solved, "
                f"with the {sky} face on top and {front} facing forward?"
            )
        else:
            user_txt = (
                f"Hey, can you move the cube so {sky} is facing up, {front} is facing forward, "
                f"and the narrow {strip} strip sits right above the red panel?"
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
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def build_montage_png(rgb_by_cam: dict[str, np.ndarray], seg_nohand_by_cam: dict[str, np.ndarray],
                      crop_size: int, crop_pad: float) -> bytes:
    views = []
    for cam in CAMS:
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
    for idx, (c, v) in enumerate(zip(CAMS, views)):
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

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = args_cli.env_spacing
    env_cfg.scene.clone_in_fabric = False
    env_cfg.seed = args_cli.seed
    env_cfg.sim.render_interval = 10**9
    env_cfg.sim.render.antialiasing_mode = args_cli.antialiasing
    config = agent_cfg["params"]["config"]
    config["num_actors"] = args_cli.num_envs
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
    n = u.num_envs
    stage = u.sim.stage
    if args_cli.light_intensity is not None or args_cli.light_color is not None:
        dome = UsdLux.DomeLight(stage.GetPrimAtPath("/World/Light"))
        if args_cli.light_intensity is not None:
            dome.GetIntensityAttr().Set(float(args_cli.light_intensity))
        if args_cli.light_color is not None:
            dome.GetColorAttr().Set(Gf.Vec3f(*[float(c) for c in args_cli.light_color]))
    hands = [UsdGeom.Imageable(stage.GetPrimAtPath(f"/World/envs/env_{i}/Robot")) for i in range(n)]
    orient_quats_t = torch.tensor(ORIENT_QUATS, dtype=torch.float32, device=device)

    first_render = [True]

    def render_montages(active_eids: list[int]) -> dict[int, bytes]:
        n_warm = args_cli.warmup_renders + (args_cli.first_warmup_renders if first_render[0] else 0)
        first_render[0] = False
        for _ in range(n_warm):
            u.sim.render()
        rgb_np = {}
        for cam in CAMS:
            sensor = u.z1_sensors[cam]
            sensor.update(0.0, force_recompute=True)
            rgb_np[cam] = sensor.data.output["rgb"][..., :3].detach().cpu().numpy()
        for h in hands:
            h.MakeInvisible()
        for _ in range(args_cli.warmup_renders):
            u.sim.render()
        seg_np = {}
        for cam in CAMS:
            sensor = u.z1_sensors[cam]
            sensor.update(0.0, force_recompute=True)
            lut = label_lut(sensor.data.info.get("semantic_segmentation") if isinstance(sensor.data.info, dict)
                            else sensor.data.info, device)
            raw = sensor.data.output["semantic_segmentation"][..., 0].long().clamp(min=0, max=len(lut) - 1)
            seg_np[cam] = lut[raw].to(torch.uint8).detach().cpu().numpy()
        for h in hands:
            h.MakeVisible()
        return {
            eid: build_montage_png(
                {c: rgb_np[c][eid] for c in CAMS},
                {c: seg_np[c][eid] for c in CAMS},
                args_cli.crop_size,
                args_cli.crop_pad,
            )
            for eid in active_eids
        }

    def read_states() -> list[tuple[int, int]]:
        quats = u.object_rot.detach().cpu().numpy()
        hinges = u.object.data.joint_pos[:, 0].detach().cpu().numpy()
        out = []
        for i in range(n):
            oid, _ = nearest_orientation(quats[i])
            det, _ = rad_to_detent(float(hinges[i]))
            out.append((oid, det))
        return out

    def current_obs() -> torch.Tensor:
        u._compute_intermediate_values()
        return torch.clamp(u.compute_full_observations(), -clip_obs, clip_obs)

    def set_goals(env_ids: list[int], actions: list[tuple[str, int]]) -> None:
        if not env_ids:
            return
        ids_t = torch.tensor(env_ids, dtype=torch.long, device=device)
        cur_hinge = u.object.data.joint_pos[ids_t, 0]
        cur_det_rad = _wrap(torch.round(cur_hinge / QUARTER_TURN) * QUARTER_TURN)
        for idx, (eid, (kind, val)) in enumerate(zip(env_ids, actions)):
            if kind == "reorient":
                u.goal_rot[eid] = orient_quats_t[int(val)]
                u.hinge_goal[eid] = cur_det_rad[idx]
                u.kind[eid] = UP_KIND if ORIENTATIONS[int(val)]["is_top_up"] else ANY_KIND
            else:
                oid, _ = nearest_orientation(u.object_rot[eid].detach().cpu().numpy())
                u.goal_rot[eid] = orient_quats_t[oid]
                u.hinge_goal[eid] = _wrap(cur_det_rad[idx] + float(val) * QUARTER_TURN)
                u.kind[eid] = TURN_KIND
        u.phase_steps[ids_t] = 0
        u.hold_count[ids_t] = 0
        u.max_hold_count[ids_t] = 0
        u.phase_start_hinge[ids_t] = cur_hinge
        u.reset_goal_buf[ids_t] = 0
        u._sync_prev_dists(ids_t)

    @torch.no_grad()
    def step_v4(obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s_rot, s_h, s_k = u.goal_rot.clone(), u.hinge_goal.clone(), u.kind.clone()
        act = player.get_action(player.obs_to_torch(obs), is_deterministic=True)
        env.step(act)
        goal_dist = torch.norm(u.object_pos - u.in_hand_pos, p=2, dim=-1)
        fallen = (goal_dist >= u.cfg.fall_dist) | u.reset_terminated.bool()
        u.goal_rot[:] = s_rot
        u.hinge_goal[:] = s_h
        u.kind[:] = s_k
        u.reset_goal_buf[:] = 0
        return current_obs(), fallen

    rng = np.random.default_rng(args_cli.seed)
    levels = [int(x) for x in args_cli.levels.split(",")]
    all_episodes = []
    step_stats = collections.Counter()
    t0 = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(n, 16)) as pool:
        for L in levels:
            for b_idx in range(args_cli.batches_per_level):
                env.reset()
                # Settle into clean discrete starting orientations.
                # Note: distance L=4 only exists from non-top-up orientations (from top-up, max BFS distance is 3).
                if L >= 4:
                    non_top = [o["id"] for o in ORIENTATIONS if not o["is_top_up"]]
                    init_oids = [int(rng.choice(non_top)) for _ in range(n)]
                else:
                    init_oids = [int(rng.integers(0, 24)) for _ in range(n)]
                set_goals(list(range(n)), [("reorient", o) for o in init_oids])
                obs = current_obs()
                for _ in range(args_cli.settle_steps):
                    obs, _ = step_v4(obs)

                cur_states = read_states()
                dst_states = []
                for oid, det in cur_states:
                    sidx = state_to_index(oid, det)
                    cands = np.nonzero(DIST_TABLE[sidx] == L)[0]
                    if len(cands) == 0:
                        cands = np.nonzero(DIST_TABLE[sidx] == int(DIST_TABLE[sidx].max()))[0]
                    dst_idx = int(rng.choice(cands))
                    dst_states.append(index_to_state(dst_idx))

                max_rounds = 2 * L + 2
                done = [False] * n
                success = [False] * n
                fell_any = [False] * n
                rounds_used = [0] * n
                ctrl_steps_used = [0] * n
                ep_logs = [[] for _ in range(n)]

                for rnd in range(1, max_rounds + 1):
                    active = [i for i in range(n) if not done[i]]
                    if not active:
                        break
                    cur_states = read_states()
                    pngs = render_montages(active)
                    futs = {}
                    for eid in active:
                        dst_o, dst_d = dst_states[eid]
                        dst_dict = {
                            "sky_face": ORIENTATIONS[dst_o]["up_colour"],
                            "front_face": ORIENTATIONS[dst_o]["front_colour"],
                            "on_red_panel_strip": on_red_strip(dst_d),
                        }
                        futs[eid] = pool.submit(
                            query_vllm_one, args_cli.vllm_url, args_cli.vllm_model,
                            pngs[eid], dst_dict, args_cli.max_tokens, args_cli.prompt_style
                        )

                    exec_ids = []
                    exec_acts = []
                    for eid in active:
                        txt = futs[eid].result()
                        obj = extract_json(txt)
                        pred_st = parse_pred_state(obj.get("state") if isinstance(obj, dict) else None)
                        act_field = obj.get("action") if isinstance(obj, dict) else None
                        pred_act = parse_action(act_field if isinstance(act_field, dict) else obj)
                        state_ok = pred_st == cur_states[eid]
                        rounds_used[eid] += 1
                        step_stats["total_rounds"] += 1
                        step_stats["state_ok"] += int(state_ok)

                        legal = False
                        if pred_act is not None:
                            if pred_act[0] == "reorient":
                                legal = True
                            elif pred_act[0] == "turn" and ORIENTATIONS[cur_states[eid][0]]["is_top_up"]:
                                legal = True
                        if legal:
                            exec_ids.append(eid)
                            exec_acts.append(pred_act)
                        else:
                            step_stats["illegal_or_parse_fail"] += 1
                        ep_logs[eid].append({
                            "round": rnd, "cur": cur_states[eid], "pred_state": pred_st,
                            "pred_act": pred_act, "state_ok": state_ok, "legal": legal,
                        })

                    if exec_ids:
                        set_goals(exec_ids, exec_acts)
                        obs = current_obs()
                        # Step physics at 60 Hz until all exec_ids reach goal, fall, or timeout
                        act_map = dict(zip(exec_ids, exec_acts))
                        reached = {eid: False for eid in exec_ids}
                        hold_cnt = {eid: 0 for eid in exec_ids}
                        for step_i in range(1, args_cli.max_round_steps + 1):
                            obs, fallen = step_v4(obs)
                            od = torch.abs(rotation_distance(u.object_rot, u.goal_rot))
                            hd = torch.abs(u._hinge_error())
                            for eid in exec_ids:
                                if reached[eid] or fell_any[eid]:
                                    continue
                                ctrl_steps_used[eid] += 1
                                if bool(fallen[eid].item()):
                                    fell_any[eid] = True
                                    done[eid] = True
                                    continue
                                if act_map[eid][0] == "turn":
                                    within = (float(od[eid]) <= u.turn_orient_tol) and (float(hd[eid]) <= u.cfg.success_tolerance)
                                    hold_cnt[eid] = hold_cnt[eid] + 1 if within else 0
                                    if hold_cnt[eid] >= u.hold_steps:
                                        reached[eid] = True
                                else:
                                    if float(od[eid]) <= u.cfg.success_tolerance and float(hd[eid]) <= u.cfg.success_tolerance:
                                        reached[eid] = True
                            if all(reached[eid] or fell_any[eid] for eid in exec_ids):
                                break

                    post_states = read_states()
                    for eid in active:
                        if fell_any[eid]:
                            done[eid] = True
                        elif post_states[eid] == dst_states[eid]:
                            success[eid] = True
                            done[eid] = True
                        elif rounds_used[eid] >= max_rounds:
                            done[eid] = True

                for eid in range(n):
                    all_episodes.append({
                        "L": L, "batch": b_idx, "env": eid,
                        "dst": dst_states[eid], "success": success[eid], "fell": fell_any[eid],
                        "rounds": rounds_used[eid], "ctrl_steps": ctrl_steps_used[eid],
                        "history": ep_logs[eid],
                    })
                print(
                    f"isaac_closedloop L={L} batch={b_idx} succ={sum(success)}/{n} "
                    f"falls={sum(fell_any)}/{n} elapsed={time.time()-t0:.1f}s",
                    flush=True,
                )

    def agg(eps: list[dict]) -> dict:
        m = max(len(eps), 1)
        succ = [e for e in eps if e["success"]]
        return {
            "n": len(eps),
            "success_rate": round(len(succ) / m, 4),
            "fall_rate": round(sum(1 for e in eps if e["fell"]) / m, 4),
            "mean_rounds_ratio": round(sum(e["rounds"] / e["L"] for e in succ) / len(succ), 4) if succ else None,
            "mean_ctrl_steps_per_round": round(
                sum(e["ctrl_steps"] for e in eps) / max(sum(e["rounds"] for e in eps), 1), 1
            ),
        }

    nr = max(step_stats["total_rounds"], 1)
    summary = {
        "model": args_cli.vllm_model,
        "checkpoint": str(checkpoint),
        "seconds": round(time.time() - t0, 1),
        "overall": agg(all_episodes),
        "by_L": {f"L{L}": agg([e for e in all_episodes if e["L"] == L]) for L in levels},
        "round_metrics": {
            "total_rounds": step_stats["total_rounds"],
            "live_state_exact_96": round(step_stats["state_ok"] / nr, 4),
            "illegal_or_parse_fail": round(step_stats["illegal_or_parse_fail"] / nr, 4),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "episodes.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in all_episodes), encoding="utf-8"
    )
    print("isaac_closedloop_summary=" + json.dumps(summary), flush=True)
    print("eval_c2_isaac_closedloop_status=PASS", flush=True)


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
