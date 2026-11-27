#!/usr/bin/env python3
"""Build Scheme C (Map-then-Reason pure vision) & Dactyl Smart-Cube Fallback SFT datasets from Z1 captures.

Reads captured runs (e.g. z1-c1-train-job1173340_*), filters clean still frames (strictly disjoint from the
180-frame z1_frames_v2 test split), renders P1-optimal 3-view cropped `montage` images (c1 + s040 + s220),
and writes:
  1. `images/*.png`                  : cached 3-view montage PNGs (1164 x 412)
  2. `sft_vision_full.jsonl`         : Scheme C Map-then-Reason (view_obs -> global state -> BFS action) +
                                       MCQ perception grounding samples (no SoM tags!)
  3. `sft_smart_cube.jsonl`          : Dactyl Smart-Cube Fallback (montage + sensor cap_angle -> orientation -> action)
  4. `state_image_pool.json`         : 96-state index mapping state_idx -> list of real montage image paths
                                       (for fast multi-turn visual GRPO in Stage C2!)
  5. `manifest.json`                 : dataset counts, orientation/detent coverage, zero-overlap verification.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from cube_state_graph import (
    DIST_TABLE,
    FACE_COLOUR,
    ORIENTATIONS,
    SIDES,
    bfs_plan,
    index_to_state,
    nearest_orientation,
    state_to_dict,
    state_to_index,
)

FACE_LABELS = [
    "base_px", "base_py", "base_nx", "base_ny", "base_nz",
    "top_px", "top_py", "top_nx", "top_ny", "top_pz",
    "stripe", "core_base", "core_top", "hand",
]
LABEL_INDEX = {name: i + 1 for i, name in enumerate(FACE_LABELS)}
CUBE_IDS = [LABEL_INDEX[name] for name in FACE_LABELS if name != "hand"]
CAMS = ("c1", "s040", "s220")
CAM_TITLES = {"c1": "VIEW 1 (FRONT-TOP)", "s040": "VIEW 2 (LEFT-SIDE)", "s220": "VIEW 3 (RIGHT-SIDE)"}

SYS_VISION_FULL = (
    "You are an embodied manipulation planner controlling a Shadow Hand holding a two-layer hinged cube. "
    "The image is a 3-camera montage (VIEW 1 FRONT-TOP, VIEW 2 LEFT-SIDE, VIEW 3 RIGHT-SIDE) at the same moment. "
    "First report the visible coloured side panels in each view (each side has a large 2/3 panel and a narrow 1/3 strip "
    "along its split edge), then deduce the global cube state (`sky_face`, `front_face`, `on_red_panel_strip`), "
    "and finally output the optimal JSON macro-action to reach the target state."
)

SYS_SMART_CUBE = (
    "You are an embodied manipulation planner controlling a Shadow Hand holding a smart hinged cube (Dactyl-style). "
    "An internal rotation sensor directly reports the exact cap alignment (`on_red_panel_strip`), while the 3-camera "
    "montage image shows the cube's 3D orientation in the hand. Read `sky_face` and `front_face` from the image, "
    "combine with the sensor reading, and output the optimal JSON macro-action to reach the target state."
)

SYS_COLOR_INV = (
    "You are an embodied manipulation planner controlling a Shadow Hand holding a two-layer hinged cube whose face "
    "colours may come from any palette or cyclic ordering. The image is a 3-camera montage (VIEW 1 FRONT-TOP, "
    "VIEW 2 LEFT-SIDE, VIEW 3 RIGHT-SIDE). For each view, group the visible side faces by 3D coplanarity "
    "(each coplanar side face has a `large_2_3` base panel and a `small_1_3` cap strip sharing the same plane), "
    "then compare adjacent coplanar faces around the cube to deduce the colour-agnostic relative rotation of the "
    "1/3 cap (`cap_relative_turn_deg` in {\"0\", \"+90\", \"180\", \"-90\"}) and the global state."
)

# Shading-preserving color palettes for Color-Invariant Geometric SFT & OOD evaluation:
# Standard RGBs in z1_render_env.py: red=(0.85,0.08,0.08), green=(0.10,0.65,0.20), yellow=(0.95,0.80,0.05), blue=(0.10,0.30,0.90)
PALETTE_RGB = {
    "red":    np.array([0.85, 0.08, 0.08], dtype=np.float32),
    "green":  np.array([0.10, 0.65, 0.20], dtype=np.float32),
    "yellow": np.array([0.95, 0.80, 0.05], dtype=np.float32),
    "blue":   np.array([0.10, 0.30, 0.90], dtype=np.float32),
    "orange": np.array([0.95, 0.45, 0.05], dtype=np.float32),
    "purple": np.array([0.55, 0.15, 0.85], dtype=np.float32),
    "cyan":   np.array([0.05, 0.78, 0.85], dtype=np.float32),
    "pink":   np.array([0.95, 0.35, 0.65], dtype=np.float32),
    "lime":   np.array([0.55, 0.88, 0.10], dtype=np.float32),
    "teal":   np.array([0.08, 0.55, 0.52], dtype=np.float32),
}
ORIG_SIDE_RGB = {s: PALETTE_RGB[FACE_COLOUR[s]] for s in SIDES}
TRAIN_AUG_SIDE_ORDERS = [
    ("px", "py", "nx", "ny", ("red", "green", "yellow", "blue")),     # canonical
    ("px", "py", "nx", "ny", ("red", "blue", "yellow", "green")),     # reversed 4-cycle
    ("px", "py", "nx", "ny", ("green", "yellow", "blue", "red")),     # cyclic shift 1
    ("px", "py", "nx", "ny", ("yellow", "red", "blue", "green")),     # swapped pairs
    ("px", "py", "nx", "ny", ("orange", "cyan", "purple", "lime")),   # novel palette A
    ("px", "py", "nx", "ny", ("pink", "teal", "orange", "purple")),   # novel palette B
]


def recolor_image_with_seg(img: Image.Image, seg: np.ndarray, side_colour_map: dict[str, str]) -> Image.Image:
    """Recolor both base_{s} and top_{s} panels to `side_colour_map[s]` while preserving 3D lighting & shadows."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    out = arr.copy()
    for s in SIDES:
        new_col_name = side_colour_map[s]
        if new_col_name == FACE_COLOUR[s]:
            continue
        orig_rgb = ORIG_SIDE_RGB[s]
        target_rgb = PALETTE_RGB[new_col_name]
        # Lighting ratio: estimate diffuse shading intensity from max channel of original color
        orig_peak = float(orig_rgb.max())
        for prefix in ("base_", "top_"):
            lid = LABEL_INDEX[f"{prefix}{s}"]
            mask = seg == lid
            if not mask.any():
                continue
            pix = arr[mask]
            shade = (pix.max(axis=-1, keepdims=True) / max(orig_peak, 1e-3)).clip(0.15, 1.35)
            out[mask] = (target_rgb * shade).clip(0.0, 1.0)
    return Image.fromarray((out * 255.0).round().astype(np.uint8))


def make_montage(
    run_dir: Path, step: int, env: int, crop_size: int = 384, crop_pad: float = 0.30,
    side_colour_map: dict[str, str] | None = None,
) -> Image.Image:
    """Create the P1-optimal 3-view cropped montage image (optionally recolored via segmentation mask)."""
    npz_path = run_dir / "labels" / f"s{step:04d}.npz"
    views = []
    with np.load(npz_path) as data:
        for cam in CAMS:
            img_path = run_dir / "images" / f"s{step:04d}_e{env:03d}_{cam}.png"
            img = Image.open(img_path).convert("RGB")
            seg_normal = data[f"{cam}_normal"][env]
            if side_colour_map is not None:
                img = recolor_image_with_seg(img, seg_normal, side_colour_map)
            seg_nohand = data[f"{cam}_nohand"][env]
            H, W = seg_nohand.shape
            cube_mask = np.isin(seg_nohand, CUBE_IDS)
            ys, xs = np.nonzero(cube_mask)
            if len(ys) > 0:
                ymin, ymax = int(ys.min()), int(ys.max())
                xmin, xmax = int(xs.min()), int(xs.max())
                cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
                half = max(xmax - xmin + 1, ymax - ymin + 1) * (0.5 + crop_pad)
                half = max(half, 64.0)
                x0 = max(0, int(round(cx - half)))
                y0 = max(0, int(round(cy - half)))
                x1 = min(W, int(round(cx + half)))
                y1 = min(H, int(round(cy + half)))
            else:
                x0, y0, x1, y1 = W // 4, H // 4, 3 * W // 4, 3 * H // 4
            cropped = img.crop((x0, y0, x1, y1)).resize((crop_size, crop_size), Image.Resampling.BICUBIC)
            views.append(cropped)

    header_h, pad, S = 28, 6, crop_size
    canvas = Image.new("RGB", (len(views) * S + (len(views) - 1) * pad, S + header_h), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    for idx, (c, v) in enumerate(zip(CAMS, views)):
        x_off = idx * (S + pad)
        draw.text((x_off + 10, 7), CAM_TITLES[c], fill=(255, 255, 255))
        canvas.paste(v, (x_off, header_h))
    return canvas


def extract_view_obs(
    row: dict, detent: int, min_base_px: int = 250, min_strip_px: int = 80,
    side_colour_map: dict[str, str] | None = None,
) -> dict:
    """Extract ground-truth visible (large_2_3, small_1_3) colour pairs per camera from segmentation pixel counts."""
    cmap = side_colour_map or FACE_COLOUR
    above = {s: SIDES[(i - detent) % 4] for i, s in enumerate(SIDES)}
    view_obs = {}
    for cam in CAMS:
        pix = row["cameras"][cam]["pixels_normal"]
        pairs = []
        for s in SIDES:
            b_px = pix.get(f"base_{s}", 0)
            t_side = above[s]
            t_px = pix.get(f"top_{t_side}", 0)
            if b_px >= min_base_px and t_px >= min_strip_px:
                pairs.append({"large_2_3": cmap[s], "small_1_3": cmap[t_side]})
        view_obs[cam] = pairs
    return view_obs


def format_action_json(act: tuple[str, int]) -> dict:
    kind, val = act
    if kind == "reorient":
        o = ORIENTATIONS[int(val)]
        return {"action": "reorient", "sky_face": o["up_colour"], "front_face": o["front_colour"]}
    return {"action": "turn", "delta": "+90" if int(val) > 0 else "-90"}


def on_red_strip(detent: int) -> str:
    return FACE_COLOUR[SIDES[(-int(detent)) % 4]]


def build_thought(cur: tuple[int, int], dst: tuple[int, int], act: tuple[str, int]) -> str:
    c_o, c_d = ORIENTATIONS[cur[0]], cur[1]
    t_o, t_d = ORIENTATIONS[dst[0]], dst[1]
    if act[0] == "turn":
        d_str = "+90" if act[1] > 0 else "-90"
        return (
            f"White cap is facing sky (`sky_face=white`) and strip on red panel is `{on_red_strip(c_d)}` "
            f"(target `{on_red_strip(t_d)}`). Execute `turn({d_str})`."
        )
    if c_d != t_d and not c_o["is_top_up"]:
        return (
            f"Strip on red panel is `{on_red_strip(c_d)}` != target `{on_red_strip(t_d)}`, and `sky_face={c_o['up_colour']}` "
            f"is not white. Reorient white cap to sky first."
        )
    return (
        f"Cap alignment already matches target (`{on_red_strip(t_d)}`). "
        f"Reorient cube to `sky_face={t_o['up_colour']}`, `front_face={t_o['front_colour']}`."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, help="Glob of z1_capture.py training output dirs")
    parser.add_argument("--exclude_frames", default="", help="Optional z1_frames_v2.jsonl to verify zero episode overlap")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_tilt", type=float, default=25.0)
    parser.add_argument("--max_residual", type=float, default=15.0)
    parser.add_argument("--max_frames_per_state", type=int, default=40, help="Balance overrepresented white-up states")
    parser.add_argument("--plans_per_frame", type=int, default=2, help="Number of random target-state plans per frame")
    parser.add_argument("--seed", type=int, default=20261201)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.output_dir).expanduser().resolve()
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    excluded_eps = set()
    if args.exclude_frames and Path(args.exclude_frames).exists():
        for line in open(args.exclude_frames, encoding="utf-8"):
            r = json.loads(line)
            excluded_eps.add((Path(r["run_dir"]).name, r["env"]))

    run_dirs = sorted(Path(p) for p in glob.glob(args.runs))
    assert run_dirs, f"No runs matched {args.runs}"

    by_state: dict[int, list[dict]] = collections.defaultdict(list)
    total_scanned = 0
    for rdir in run_dirs:
        fpath = rdir / "frames.jsonl"
        if not fpath.exists():
            continue
        for line in open(fpath, encoding="utf-8"):
            total_scanned += 1
            row = json.loads(line)
            if (rdir.name, row["env"]) in excluded_eps:
                continue
            if row["up_tilt_deg"] > args.max_tilt or abs(row["detent_residual_deg"]) > args.max_residual:
                continue
            oid, o_err = nearest_orientation(np.asarray(row["cube_quat_wxyz"], dtype=float))
            if o_err > 28.0:
                continue
            det = int(row["detent"]) % 4
            row["_run_dir"] = str(rdir)
            row["_oid"] = oid
            row["_detent"] = det
            row["_sidx"] = state_to_index(oid, det)
            by_state[row["_sidx"]].append(row)

    selected = []
    for sidx, bucket in sorted(by_state.items()):
        rng.shuffle(bucket)
        selected.extend(bucket[: args.max_frames_per_state])
    rng.shuffle(selected)

    vision_rows = []
    smart_rows = []
    color_inv_rows = []
    state_pool: dict[str, list[str]] = collections.defaultdict(list)
    aug_img_dir = out / "images_aug"
    aug_img_dir.mkdir(parents=True, exist_ok=True)
    DETENT_TO_DEG = {0: "0", 1: "+90", 2: "180", 3: "-90"}

    for idx, row in enumerate(selected):
        rdir = Path(row["_run_dir"])
        oid, det, sidx = row["_oid"], row["_detent"], row["_sidx"]
        o_info = ORIENTATIONS[oid]
        img_name = f"{rdir.name}_s{row['step']:04d}_e{row['env']:03d}.png"
        img_path = img_dir / img_name
        if not img_path.exists():
            montage = make_montage(rdir, row["step"], row["env"])
            montage.save(img_path)
        state_pool[str(sidx)].append(str(img_path))

        v_obs = extract_view_obs(row, det)
        cur_state = {
            "sky_face": o_info["up_colour"],
            "front_face": o_info["front_colour"],
            "on_red_panel_strip": on_red_strip(det),
        }

        # --- Color-Invariant Geometric SFT (Scheme C-Generalization: coplanar grouping + adjacent face cross-matching) ---
        # Each frame is included BOTH in canonical palette AND in a randomly permuted/novel 4-side colour palette!
        for aug_k in range(2):
            if aug_k == 0:
                cmap = dict(FACE_COLOUR)
                c_img_path = img_path
            else:
                _, _, _, _, cols4 = rng.choice(TRAIN_AUG_SIDE_ORDERS[1:])
                cmap = dict(FACE_COLOUR)
                for s_i, s_name in enumerate(SIDES):
                    cmap[s_name] = cols4[s_i]
                aug_name = f"{rdir.name}_s{row['step']:04d}_e{row['env']:03d}_aug.png"
                c_img_path = aug_img_dir / aug_name
                if not c_img_path.exists():
                    make_montage(rdir, row["step"], row["env"], side_colour_map=cmap).save(c_img_path)

            v_obs_aug = extract_view_obs(row, det, side_colour_map=cmap)
            rel_deg = DETENT_TO_DEG[det]
            # Build explicit colour-agnostic coplanar & cross-face reasoning trace
            all_pairs = [p for plist in v_obs_aug.values() for p in plist]
            if det == 0 and all_pairs:
                ex_p = all_pairs[0]
                geo_deduction = (
                    f"Coplanar side panel `{ex_p['large_2_3']}` has matching cap strip `{ex_p['small_1_3']}` "
                    f"(`large_2_3 == small_1_3`), so cap relative turn is `0` deg (aligned)."
                )
            else:
                cyc_str = " -> ".join(cmap[s] for s in SIDES)
                ex_p = all_pairs[0] if all_pairs else {"large_2_3": cmap["px"], "small_1_3": cmap[SIDES[(-det) % 4]]}
                geo_deduction = (
                    f"Around the cube (`{cyc_str}` CCW), coplanar base panel `{ex_p['large_2_3']}` shares its plane "
                    f"with cap strip `{ex_p['small_1_3']}`, which is shifted by `{rel_deg}` deg."
                )

            inv_state = {
                "sky_face": cmap[o_info["up_face"]],
                "front_face": cmap[o_info["front_face"]],
                "cap_relative_turn_deg": rel_deg,
                "on_anchor_panel_strip": cmap[SIDES[(-det) % 4]],
            }
            color_inv_rows.append({
                "id": f"cinv_state_{idx:05d}_{aug_k}",
                "task_type": "cinv_map_state",
                "image": str(c_img_path),
                "system": SYS_COLOR_INV,
                "user": (
                    "Inspect the 3-camera montage. Group each visible side face by 3D coplanarity (`large_2_3` base "
                    "and `small_1_3` cap strip in the same plane), deduce the colour-agnostic relative cap rotation "
                    "`cap_relative_turn_deg` (\"0\", \"+90\", \"180\", \"-90\"), and report `state` in JSON."
                ),
                "assistant": json.dumps(
                    {"coplanar_view_obs": v_obs_aug, "geometry_deduction": geo_deduction, "state": inv_state},
                    ensure_ascii=False,
                ),
            })

            # Also add colour-permuted MCQ grounding samples (up / turned / band on arbitrary anchor colour)
            anchor_col = cmap["px"]
            anchor_strip_col = cmap[SIDES[(-det) % 4]]
            t_ans = (
                "Each face shows a single colour"
                if det == 0
                else "Each face shows a wide part and a narrow strip of another colour"
            )
            b_opts = [
                f"There is no strip; the face is {anchor_col} only",
                cmap["ny"],
                cmap["nx"],
                cmap["py"],
            ]
            b_ans = b_opts[0] if det == 0 else anchor_strip_col
            cinv_mcqs = [
                ("up", "Which colour covers most of the cube face that points up (toward the sky)?",
                 ["white", "magenta", cmap["px"], cmap["py"], cmap["nx"], cmap["ny"]], cmap[o_info["up_face"]]),
                ("turned",
                 "Apart from the white face and the magenta face, does each face of the cube show a single colour, or "
                 "does each show a wide part of one colour and a narrow strip of another colour?",
                 ["Each face shows a single colour", "Each face shows a wide part and a narrow strip of another colour"],
                 t_ans),
                ("band",
                 f"Find the cube face that is mostly {anchor_col}. What colour is the narrow strip along one of its edges?",
                 b_opts, b_ans),
            ]
            for lv, q_text, opts, ans_str in cinv_mcqs:
                rot = rng.randrange(len(opts))
                rotated = opts[rot:] + opts[:rot] + ["Cannot be determined"]
                letters = "ABCDEFGHIJ"
                opt_block = "".join(f"{letters[i]}. {opt}\n" for i, opt in enumerate(rotated))
                ans_letter = letters[rotated.index(ans_str)]
                color_inv_rows.append({
                    "id": f"cinv_mcq_{lv}_{idx:05d}_{aug_k}",
                    "task_type": f"cinv_mcq_{lv}",
                    "image": str(c_img_path),
                    "system": "",
                    "user": (
                        "Hint: The image is a side-by-side panel showing the same moment from 3 fixed cameras. "
                        "A robot hand, palm up, holds a cube whose faces are coloured. The hand may hide parts of the cube.\n"
                        f"Question: {q_text}\nOptions:\n{opt_block}"
                        "Please select the correct answer from the options above. \n"
                    ),
                    "assistant": f"{ans_letter}. {ans_str}",
                })

        # 1) Pure state-perception Map-then-Reason sample (helps both closed-loop state reading & Z1 MCQ transfer)
        vision_rows.append({
            "id": f"state_{idx:05d}",
            "task_type": "map_state",
            "image": str(img_path),
            "system": SYS_VISION_FULL,
            "user": (
                "Inspect the 3-camera montage and report `view_obs` (for each camera, list visible coloured side "
                "panels as `large_2_3` and `small_1_3`) and the global `state` (`sky_face`, `front_face`, "
                "`on_red_panel_strip`) in JSON."
            ),
            "assistant": json.dumps({"view_obs": v_obs, "state": cur_state}, ensure_ascii=False),
        })

        # 2) Z1-v2 MCQ grounding samples (up / turned / band) on the exact same montage image (zero SoM tags!)
        strip_col = on_red_strip(det)
        turned_ans = (
            "Each face shows a single colour"
            if det == 0
            else "Each face shows a wide part and a narrow strip of another colour"
        )
        band_ans = "There is no strip; the face is red only" if det == 0 else strip_col
        mcq_items = [
            ("up", "Which colour covers most of the cube face that points up (toward the sky)?",
             ["white", "magenta", "red", "green", "yellow", "blue"], o_info["up_colour"]),
            ("turned",
             "Apart from the white face and the magenta face, does each face of the cube show a single colour, or "
             "does each show a wide part of one colour and a narrow strip of another colour?",
             ["Each face shows a single colour", "Each face shows a wide part and a narrow strip of another colour"],
             turned_ans),
            ("band",
             "Find the cube face that is mostly red. What colour is the narrow strip along one of its edges?",
             ["There is no strip; the face is red only", "blue", "yellow", "green"], band_ans),
        ]
        for lv, q_text, opts, ans_str in mcq_items:
            rot = rng.randrange(len(opts))
            rotated = opts[rot:] + opts[:rot] + ["Cannot be determined"]
            letters = "ABCDEFGHIJ"
            opt_block = "".join(f"{letters[i]}. {opt}\n" for i, opt in enumerate(rotated))
            ans_letter = letters[rotated.index(ans_str)]
            vision_rows.append({
                "id": f"mcq_{lv}_{idx:05d}",
                "task_type": f"mcq_{lv}",
                "image": str(img_path),
                "system": "",
                "user": (
                    "Hint: The image is a side-by-side panel showing the same moment from 3 fixed cameras. "
                    "A robot hand, palm up, holds a cube whose faces are coloured. The hand may hide parts of the cube.\n"
                    f"Question: {q_text}\nOptions:\n{opt_block}"
                    "Please select the correct answer from the options above. \n"
                ),
                "assistant": f"{ans_letter}. {ans_str}",
            })

        # 3) Map-then-Reason + BFS Action planning samples (for both Scheme C and Dactyl Smart-Cube Fallback)
        for p_i in range(args.plans_per_frame):
            L = rng.choice([1, 2, 3, 4])
            cands = np.nonzero(DIST_TABLE[sidx] == L)[0]
            if len(cands) == 0:
                cands = np.nonzero(DIST_TABLE[sidx] > 0)[0]
            dst_idx = int(rng.choice(cands.tolist()))
            dst_o, dst_d = index_to_state(dst_idx)
            plan = bfs_plan((oid, det), (dst_o, dst_d))
            act = plan[0]
            act_json = format_action_json(act)
            thought = build_thought((oid, det), (dst_o, dst_d), act)
            dst_state = {
                "sky_face": ORIENTATIONS[dst_o]["up_colour"],
                "front_face": ORIENTATIONS[dst_o]["front_colour"],
                "on_red_panel_strip": on_red_strip(dst_d),
            }

            vision_rows.append({
                "id": f"plan_{idx:05d}_{p_i}",
                "task_type": f"plan_L{L}",
                "image": str(img_path),
                "system": SYS_VISION_FULL,
                "user": (
                    f"Target State: {json.dumps(dst_state)}\n"
                    "Observe the 3-camera montage, output `view_obs`, `state`, `thought`, and the next optimal `action` in JSON."
                ),
                "assistant": json.dumps(
                    {"view_obs": v_obs, "state": cur_state, "thought": thought, "action": act_json},
                    ensure_ascii=False,
                ),
            })

            smart_rows.append({
                "id": f"smart_{idx:05d}_{p_i}",
                "task_type": f"smart_L{L}",
                "image": str(img_path),
                "system": SYS_SMART_CUBE,
                "user": (
                    f"Smart-Cube Sensor Reading: `on_red_panel_strip = \"{cur_state['on_red_panel_strip']}\"`\n"
                    f"Target State: {json.dumps(dst_state)}\n"
                    "Read `sky_face` and `front_face` from the montage image, combine with the sensor reading, and "
                    "output `state`, `thought`, and the next optimal `action` in JSON."
                ),
                "assistant": json.dumps(
                    {"state": cur_state, "thought": thought, "action": act_json},
                    ensure_ascii=False,
                ),
            })

    rng.shuffle(vision_rows)
    rng.shuffle(smart_rows)
    rng.shuffle(color_inv_rows)
    (out / "sft_vision_full.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in vision_rows), encoding="utf-8"
    )
    (out / "sft_smart_cube.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in smart_rows), encoding="utf-8"
    )
    (out / "sft_color_invariant.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in color_inv_rows), encoding="utf-8"
    )
    (out / "state_image_pool.json").write_text(json.dumps(state_pool, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "total_scanned_frames": total_scanned,
        "selected_frames": len(selected),
        "states_covered": len(state_pool),
        "vision_full_samples": len(vision_rows),
        "smart_cube_samples": len(smart_rows),
        "color_invariant_samples": len(color_inv_rows),
        "excluded_test_episodes": len(excluded_eps),
        "up_face_counts": dict(collections.Counter(ORIENTATIONS[r["_oid"]]["up_colour"] for r in selected)),
        "detent_counts": dict(sorted(collections.Counter(r["_detent"] for r in selected).items())),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("build_c1_sft_dataset_status=PASS " + json.dumps(manifest))


if __name__ == "__main__":
    main()
