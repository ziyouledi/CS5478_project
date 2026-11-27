#!/usr/bin/env python3
"""Build C1-v2 / C1-v2-colorinv SFT dataset combining:
  1. Multi-template & bidirectional perception MCQs (forward base->strip, inverse strip->base,
     arbitrary anchor faces, diverse natural-language paraphrases; exact v3 wording strictly held out).
  2. Visibility-grounded abstention (Cannot be determined) supervision using real occlusion from
     segmentation pixel counts + single-view camera dropout frames where the queried face is absent.
  3. Shading-preserving colour-invariant augmentation with optional highlight/pastel jitter (strictly
     excluding held-out test `novel_palette` {orange, cyan, purple, pink} in any permutation).
  4. Colour-invariant Map-then-Reason state estimation & BFS planning with explicit cyclic-order CoT
     for `turn(+90/-90)` on BOTH canonical and arbitrary 4-colour palettes (strictly excluding the
     200 frozen test (src, dst) pairs and all 109 test episodes).
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

from build_c1_sft_dataset import (
    CAMS,
    CAM_TITLES,
    CUBE_IDS,
    LABEL_INDEX,
    ORIG_SIDE_RGB,
    PALETTE_RGB,
    SYS_VISION_FULL,
    TRAIN_AUG_SIDE_ORDERS,
    extract_view_obs,
    format_action_json,
    make_montage,
    on_red_strip,
    recolor_image_with_seg,
)
from cube_state_graph import (
    DIST_TABLE,
    FACE_COLOUR,
    ORIENTATIONS,
    SIDES,
    bfs_plan,
    index_to_state,
    nearest_orientation,
    state_to_index,
)

CBD = "Cannot be determined"
LETTERS = "ABCDEFGHIJ"
CANON_SIDES = ("red", "green", "yellow", "blue")
HELD_OUT_NOVEL_SET = frozenset({"orange", "cyan", "purple", "pink"})
AUG_COLOUR_POOL = tuple(PALETTE_RGB.keys())


def sample_aug_palette(rng: random.Random) -> dict[str, str]:
    """Sample a random non-canonical 4-colour side palette, strictly excluding held-out novel_palette."""
    if rng.random() < 0.25:
        # Non-canonical permutation of canonical 4 colours (includes reverse_cycle, cyclic shifts, swaps)
        cols4 = list(CANON_SIDES)
        while tuple(cols4) == CANON_SIDES:
            rng.shuffle(cols4)
    else:
        while True:
            cols4 = rng.sample(AUG_COLOUR_POOL, 4)
            if tuple(cols4) == CANON_SIDES:
                continue
            if frozenset(cols4) == HELD_OUT_NOVEL_SET:
                continue
            break
    assert frozenset(cols4) != HELD_OUT_NOVEL_SET, "Held-out novel_palette leaked into training!"
    cmap = dict(FACE_COLOUR)
    for s_i, s_name in enumerate(SIDES):
        cmap[s_name] = cols4[s_i]
    return cmap


def recolor_image_with_seg_jitter(
    img: Image.Image, seg: np.ndarray, side_colour_map: dict[str, str], white_blend: float = 0.0,
) -> Image.Image:
    """Recolor side faces preserving 3D shading, with optional highlight/pastel white_blend (0..0.35)."""
    if white_blend <= 1e-6:
        return recolor_image_with_seg(img, seg, side_colour_map)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    out = arr.copy()
    wb = float(white_blend)
    for s in SIDES:
        new_col_name = side_colour_map[s]
        orig_rgb = ORIG_SIDE_RGB[s]
        base_rgb = PALETTE_RGB[new_col_name]
        # Mild blend toward white simulates bright dome-light / linear-to-sRGB pastel washout (e.g. pale pink)
        target_rgb = (1.0 - wb) * base_rgb + wb * np.ones(3, dtype=np.float32)
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


def make_montage_aug(
    run_dir: Path, step: int, env: int, side_colour_map: dict[str, str],
    white_blend: float = 0.0, crop_size: int = 384, crop_pad: float = 0.30,
) -> Image.Image:
    """Create 3-view cropped montage with arbitrary side_colour_map and optional highlight/pastel jitter."""
    if white_blend <= 1e-6:
        return make_montage(run_dir, step, env, crop_size=crop_size, crop_pad=crop_pad, side_colour_map=side_colour_map)
    npz_path = run_dir / "labels" / f"s{step:04d}.npz"
    views = []
    with np.load(npz_path) as data:
        for cam in CAMS:
            img_path = run_dir / "images" / f"s{step:04d}_e{env:03d}_{cam}.png"
            img = Image.open(img_path).convert("RGB")
            seg_normal = data[f"{cam}_normal"][env]
            img = recolor_image_with_seg_jitter(img, seg_normal, side_colour_map, white_blend=white_blend)
            seg_nohand = data[f"{cam}_nohand"][env]
            H, W = seg_nohand.shape
            ys, xs = np.nonzero(np.isin(seg_nohand, CUBE_IDS))
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
    return canvas


def strip_on_anchor(detent: int, cmap: dict[str, str]) -> str:
    return cmap[SIDES[(-int(detent)) % 4]]


def format_action_json_cmap(act: tuple[str, int], cmap: dict[str, str]) -> dict:
    kind, val = act
    if kind == "reorient":
        o = ORIENTATIONS[int(val)]
        return {"action": "reorient", "sky_face": cmap[o["up_face"]], "front_face": cmap[o["front_face"]]}
    return {"action": "turn", "delta": "+90" if int(val) > 0 else "-90"}

# Paraphrase templates for forward (base->strip), inverse (strip->base), up, and turned.
# NOTE: The exact `v3` benchmark wording in `z1_mcq_eval.py` is deliberately NOT included here so v3 stays a true OOD template!
FWD_BAND_TEMPLATES = [
    "Find the cube face that is mostly {anchor}. What colour is the narrow strip along one of its edges?",
    "Locate the large {anchor} side panel (the 2/3 lower section). Which colour appears on the thin 1/3 cap strip right next to it on the same face?",
    "On the side face whose main 2/3 panel is {anchor}, what is the colour of the thin 1/3 strip sharing that face?",
]
INV_BAND_TEMPLATES = [
    "Locate the thin 1/3 {anchor} strip on the rotatable cap. Which colour is the larger 2/3 base panel sitting directly beneath that {anchor} strip?",
    "On the rotatable cap, find the narrow {anchor} strip (which sits on the side of the white cap). What colour is the wide 2/3 side panel aligned directly under it?",
    "The top 1/3 cap has a narrow {anchor} edge strip. Which coloured 2/3 base face is that {anchor} strip currently lined up with?",
]
TURNED_TEMPLATES = [
    (
        "Apart from the white face and the magenta face, does each face of the cube show a single colour, or "
        "does each show a wide part of one colour and a narrow strip of another colour?",
        ["Each face shows a single colour", "Each face shows a wide part and a narrow strip of another colour"],
    ),
    (
        "Looking at the four coloured side faces, is the rotatable 1/3 cap currently aligned with the 2/3 base "
        "(same colour on each side face) or rotated out of alignment (different strip colour on each side face)?",
        ["Aligned: the 1/3 strips match the 2/3 side panels", "Rotated: the 1/3 strips have different colours from the 2/3 side panels"],
    ),
]
UP_TEMPLATES = [
    "Which colour covers most of the cube face that points up (toward the sky)?",
    "Which coloured face of the cube is currently facing upward toward the sky?",
]


def make_single_view_crop(run_dir: Path, step: int, env: int, cam: str, crop_size: int = 384, crop_pad: float = 0.30) -> Image.Image:
    npz_path = run_dir / "labels" / f"s{step:04d}.npz"
    img_path = run_dir / "images" / f"s{step:04d}_e{env:03d}_{cam}.png"
    img = Image.open(img_path).convert("RGB")
    with np.load(npz_path) as data:
        seg_nohand = data[f"{cam}_nohand"][env]
    H, W = seg_nohand.shape
    ys, xs = np.nonzero(np.isin(seg_nohand, CUBE_IDS))
    if len(ys) > 0:
        ymin, ymax = int(ys.min()), int(ys.max())
        xmin, xmax = int(xs.min()), int(xs.max())
        cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
        half = max(max(xmax - xmin + 1, ymax - ymin + 1) * (0.5 + crop_pad), 64.0)
        x0, y0 = max(0, int(round(cx - half))), max(0, int(round(cy - half)))
        x1, y1 = min(W, int(round(cx + half))), min(H, int(round(cy + half)))
    else:
        x0, y0, x1, y1 = W // 4, H // 4, 3 * W // 4, 3 * H // 4
    return img.crop((x0, y0, x1, y1)).resize((crop_size, crop_size), Image.Resampling.BICUBIC)


def format_mcq_row(
    rid: str, task_type: str, img_path: Path, q_text: str, opts: list[str], ans_str: str,
    rng: random.Random, n_cams: int = 3,
) -> dict:
    rot = rng.randrange(len(opts))
    rotated = opts[rot:] + opts[:rot] + [CBD]
    opt_block = "".join(f"{LETTERS[i]}. {opt}\n" for i, opt in enumerate(rotated))
    ans_letter = LETTERS[rotated.index(ans_str)]
    if n_cams > 1:
        hint = (
            f"The image is a side-by-side panel showing the same moment from {n_cams} fixed cameras. "
            "A robot hand, palm up, holds a cube whose faces are coloured. The hand may hide parts of the cube."
        )
    else:
        hint = (
            "The images show the same moment from 1 fixed camera. "
            "A robot hand, palm up, holds a cube whose faces are coloured. The hand may hide parts of the cube."
        )
    return {
        "id": rid,
        "task_type": task_type,
        "image": str(img_path),
        "system": "",
        "user": (
            f"Hint: {hint}\n"
            f"Question: {q_text}\nOptions:\n{opt_block}"
            "Please select the correct answer from the options above. \n"
        ),
        "assistant": f"{ans_letter}. {ans_str}",
    }


def build_thought_v2(
    cur: tuple[int, int], dst: tuple[int, int], act: tuple[str, int], cmap: dict[str, str] | None = None,
) -> str:
    """Explicit cyclic-order CoT for canonical or arbitrary 4-colour palettes so turn(+90/-90) never confuses sign."""
    cm = cmap or FACE_COLOUR
    anchor = cm["px"]
    c_o, c_d = ORIENTATIONS[cur[0]], cur[1]
    t_o, t_d = ORIENTATIONS[dst[0]], dst[1]
    cur_s = strip_on_anchor(c_d, cm)
    tgt_s = strip_on_anchor(t_d, cm)
    plus90_cycle = f"{cm['px']}->{cm['ny']}->{cm['nx']}->{cm['py']}->{cm['px']}"
    if act[0] == "turn":
        d_str = "+90" if act[1] > 0 else "-90"
        diff = (t_d - c_d) % 4
        step_desc = "+1 step (+90)" if diff == 1 else ("-1 step (-90)" if diff == 3 else f"2 steps ({d_str} first)")
        return (
            f"White cap faces sky (`sky_face=white`). On the {anchor} panel, +90 cycle is {plus90_cycle} "
            f"(and -90 is reverse). Current strip is `{cur_s}`, target is `{tgt_s}` ({step_desc}). Execute `turn({d_str})`."
        )
    if c_d != t_d and not c_o["is_top_up"]:
        return (
            f"Strip on {anchor} panel is `{cur_s}` != target `{tgt_s}`, and `sky_face={cm[c_o['up_face']]}` "
            f"is not white. Reorient white cap to sky first so turning is legal."
        )
    return (
        f"Cap alignment already matches target (`{tgt_s}`). "
        f"Reorient cube to `sky_face={cm[t_o['up_face']]}`, `front_face={cm[t_o['front_face']]}`."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, help="Glob of z1_capture.py training output dirs")
    parser.add_argument("--exclude_frames", required=True, help="z1_frames_v2.jsonl (exclude test episodes)")
    parser.add_argument("--exclude_tasks", required=True, help="Frozen 200 test tasks.json (exclude (src,dst) pairs)")
    parser.add_argument("--reuse_img_dir", default="", help="Optional existing montage images dir to reuse PNGs")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_tilt", type=float, default=25.0)
    parser.add_argument("--max_residual", type=float, default=15.0)
    parser.add_argument("--max_frames_per_state", type=int, default=25)
    parser.add_argument("--plans_per_frame", type=int, default=3)
    parser.add_argument("--aug_prob", type=float, default=1.0, help="Fraction of frames getting arbitrary-palette augmentation")
    parser.add_argument("--aug_plans_per_frame", type=int, default=2, help="Augmented-palette BFS plans per augmented frame")
    parser.add_argument("--pastel_jitter_prob", type=float, default=0.35, help="Prob of highlight/pastel blend on aug frames")
    parser.add_argument("--min_face_px", type=int, default=300)
    parser.add_argument("--min_band_px", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261121)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.output_dir).expanduser().resolve()
    img_dir = out / "images"
    aug_dir = out / "images_aug"
    single_dir = out / "images_single"
    for d in (img_dir, aug_dir, single_dir):
        d.mkdir(parents=True, exist_ok=True)
    reuse_dir = Path(args.reuse_img_dir).expanduser().resolve() if args.reuse_img_dir else None

    excluded_eps = set()
    for line in open(args.exclude_frames, encoding="utf-8"):
        r = json.loads(line)
        excluded_eps.add((Path(r["run_dir"]).name, r["env"]))

    excluded_pairs = set()
    for t in json.loads(Path(args.exclude_tasks).read_text(encoding="utf-8")):
        s = state_to_index(int(t["src"]["orient_id"]), int(t["src"]["detent"]))
        d = state_to_index(int(t["dst"]["orient_id"]), int(t["dst"]["detent"]))
        excluded_pairs.add((s, d))

    run_dirs = sorted(Path(p) for p in glob.glob(args.runs))
    assert run_dirs, f"No runs matched {args.runs}"

    by_state: dict[int, list[dict]] = collections.defaultdict(list)
    for rdir in run_dirs:
        fpath = rdir / "frames.jsonl"
        if not fpath.exists():
            continue
        for line in open(fpath, encoding="utf-8"):
            row = json.loads(line)
            if (rdir.name, row["env"]) in excluded_eps:
                continue
            if row["up_tilt_deg"] > args.max_tilt or abs(row["detent_residual_deg"]) > args.max_residual:
                continue
            # Ensure all 3 camera PNGs and label npz exist on disk (guards against partial capture dirs)
            if not (rdir / "labels" / f"s{row['step']:04d}.npz").exists():
                continue
            if not all((rdir / "images" / f"s{row['step']:04d}_e{row['env']:03d}_{c}.png").exists() for c in CAMS):
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
    assert len(selected) > 0, f"No valid 3-camera frames found in {args.runs}!"
    rng.shuffle(selected)

    rows_out = []
    counts = collections.Counter()

    for idx, row in enumerate(selected):
        rdir = Path(row["_run_dir"])
        oid, det, sidx = row["_oid"], row["_detent"], row["_sidx"]
        o_info = ORIENTATIONS[oid]
        img_name = f"{rdir.name}_s{row['step']:04d}_e{row['env']:03d}.png"
        if reuse_dir and (reuse_dir / img_name).exists():
            img_path = reuse_dir / img_name
        else:
            img_path = img_dir / img_name
            if not img_path.exists():
                make_montage(rdir, row["step"], row["env"]).save(img_path)

        v_obs = extract_view_obs(row, det)
        cur_state = {
            "sky_face": o_info["up_colour"],
            "front_face": o_info["front_colour"],
            "on_red_panel_strip": on_red_strip(det),
        }

        # Visibility helpers across the 3 cameras (matching z1_mcq_eval.py thresholds)
        pix_list = [row["cameras"][c]["pixels_normal"] for c in CAMS]
        above_map = {s: SIDES[(i - det) % 4] for i, s in enumerate(SIDES)}

        def is_fwd_visible(anchor_s: str) -> bool:
            t_s = above_map[anchor_s]
            return any(
                p.get(f"base_{anchor_s}", 0) >= args.min_face_px and p.get(f"top_{t_s}", 0) >= args.min_band_px
                for p in pix_list
            )

        def is_inv_visible(strip_s: str) -> bool:
            # Inverse: which base face sits under `top_{strip_s}`?
            b_idx = (SIDES.index(strip_s) + det) % 4
            b_s = SIDES[b_idx]
            side_ok = any(
                p.get(f"base_{b_s}", 0) >= args.min_face_px and p.get(f"top_{strip_s}", 0) >= args.min_band_px
                for p in pix_list
            )
            bar_ok = (strip_s == "px") and any(
                p.get("top_pz", 0) >= args.min_face_px and p.get(f"base_{b_s}", 0) >= args.min_face_px
                for p in pix_list
            )
            return side_ok or bar_ok

        # 1) Canonical map_state sample
        rows_out.append({
            "id": f"v2_state_{idx:05d}",
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
        counts["map_state"] += 1

        # 2) Canonical MCQs: up + turned + forward band (visibility-gated CBD!) + inverse band (paraphrased!)
        up_q = rng.choice(UP_TEMPLATES)
        rows_out.append(format_mcq_row(
            f"v2_mcq_up_{idx:05d}", "mcq_up", img_path, up_q,
            ["white", "magenta", "red", "green", "yellow", "blue"], o_info["up_colour"], rng,
        ))
        counts["mcq_up"] += 1

        t_q, t_opts = rng.choice(TURNED_TEMPLATES)
        rows_out.append(format_mcq_row(
            f"v2_mcq_turned_{idx:05d}", "mcq_turned", img_path, t_q, t_opts,
            t_opts[0] if det == 0 else t_opts[1], rng,
        ))
        counts["mcq_turned"] += 1

        # Forward band on red (with genuine occlusion -> CBD!)
        fwd_vis = is_fwd_visible("px")
        fwd_opts = ["There is no strip; the face is red only", "blue", "yellow", "green"]
        fwd_ans = (fwd_opts[0] if det == 0 else on_red_strip(det)) if fwd_vis else CBD
        rows_out.append(format_mcq_row(
            f"v2_mcq_fwd_{idx:05d}", "mcq_fwd_cbd" if not fwd_vis else "mcq_fwd",
            img_path, rng.choice(FWD_BAND_TEMPLATES).format(anchor="red"), fwd_opts, fwd_ans, rng,
        ))
        counts["mcq_fwd_cbd" if not fwd_vis else "mcq_fwd"] += 1

        # Inverse band (strip -> base panel, paraphrased templates, random anchor 50% red / 50% other side)
        inv_anchor_s = "px" if rng.random() < 0.6 else rng.choice(SIDES)
        inv_anchor_col = FACE_COLOUR[inv_anchor_s]
        inv_vis = is_inv_visible(inv_anchor_s)
        tgt_base_col = FACE_COLOUR[SIDES[(SIDES.index(inv_anchor_s) + det) % 4]]
        inv_opts = ["red", "green", "yellow", "blue"]
        inv_ans = tgt_base_col if inv_vis else CBD
        rows_out.append(format_mcq_row(
            f"v2_mcq_inv_{idx:05d}", "mcq_inv_cbd" if not inv_vis else "mcq_inv",
            img_path, rng.choice(INV_BAND_TEMPLATES).format(anchor=inv_anchor_col), inv_opts, inv_ans, rng,
        ))
        counts["mcq_inv_cbd" if not inv_vis else "mcq_inv"] += 1

        # 3) Single-view dropout abstention sample (when a single camera cannot see red face -> genuine CBD!)
        occluded_cams = [
            c for c in CAMS
            if row["cameras"][c]["pixels_normal"].get("base_px", 0) < 120
        ]
        if occluded_cams and rng.random() < 0.55:
            oc_cam = rng.choice(occluded_cams)
            s_name = f"{rdir.name}_s{row['step']:04d}_e{row['env']:03d}_{oc_cam}.png"
            s_path = single_dir / s_name
            if not s_path.exists():
                make_single_view_crop(rdir, row["step"], row["env"], oc_cam).save(s_path)
            rows_out.append(format_mcq_row(
                f"v2_cbd_single_{idx:05d}", "mcq_cbd_single", s_path,
                FWD_BAND_TEMPLATES[0].format(anchor="red"), fwd_opts, CBD, rng, n_cams=1,
            ))
            counts["mcq_cbd_single"] += 1

        # 4) Colour-invariant augmentation: arbitrary 4-colour palettes (strictly excluding held-out novel_palette
        #    {orange, cyan, purple, pink}) for MCQs, map_state, AND Map-then-Reason BFS planning trajectories!
        if rng.random() < args.aug_prob:
            cmap = sample_aug_palette(rng)
            white_blend = rng.uniform(0.10, 0.35) if rng.random() < args.pastel_jitter_prob else 0.0
            aug_name = f"{rdir.name}_s{row['step']:04d}_e{row['env']:03d}_aug.png"
            aug_path = aug_dir / aug_name
            if not aug_path.exists():
                make_montage_aug(rdir, row["step"], row["env"], side_colour_map=cmap, white_blend=white_blend).save(aug_path)

            anchor_col = cmap["px"]
            strip_key = f"on_{anchor_col}_panel_strip"
            palette_str = f"{cmap['px']}, {cmap['py']}, {cmap['nx']}, {cmap['ny']}"
            aug_strip_col = strip_on_anchor(det, cmap)
            v_obs_aug = extract_view_obs(row, det, side_colour_map=cmap)
            cur_state_aug = {
                "sky_face": cmap[o_info["up_face"]],
                "front_face": cmap[o_info["front_face"]],
                strip_key: aug_strip_col,
            }

            # 4a) Colour-augmented MCQs (50% of augmented frames, with visibility-gated CBD)
            if rng.random() < 0.50:
                aug_fwd_opts = [f"There is no strip; the face is {anchor_col} only", cmap["ny"], cmap["nx"], cmap["py"]]
                aug_fwd_ans = (aug_fwd_opts[0] if det == 0 else aug_strip_col) if fwd_vis else CBD
                rows_out.append(format_mcq_row(
                    f"v2_aug_up_{idx:05d}", "mcq_aug_up", aug_path, rng.choice(UP_TEMPLATES),
                    ["white", "magenta", cmap["px"], cmap["py"], cmap["nx"], cmap["ny"]],
                    cmap[o_info["up_face"]], rng,
                ))
                rows_out.append(format_mcq_row(
                    f"v2_aug_fwd_{idx:05d}", "mcq_aug_fwd", aug_path,
                    rng.choice(FWD_BAND_TEMPLATES).format(anchor=anchor_col), aug_fwd_opts, aug_fwd_ans, rng,
                ))
                counts["mcq_aug"] += 2

            # 4b) Colour-augmented map_state (generalizes view_obs + sky/front/on_<anchor>_panel_strip)
            if rng.random() < 0.50:
                state_user = (
                    f"This cube uses a novel side-colour palette: {palette_str} (top is white, bottom is magenta).\n"
                    "Inspect the 3-camera montage and report `view_obs` (for each camera, list visible coloured side "
                    f"panels as `large_2_3` and `small_1_3`) and the global `state` (`sky_face`, `front_face`, "
                    f"`{strip_key}`) in JSON."
                )
            else:
                state_user = (
                    "Inspect the 3-camera montage and report `view_obs` (for each camera, list visible coloured side "
                    f"panels as `large_2_3` and `small_1_3`) and the global `state` (`sky_face`, `front_face`, "
                    f"`{strip_key}`) in JSON."
                )
            rows_out.append({
                "id": f"v2_state_aug_{idx:05d}",
                "task_type": "map_state_aug",
                "image": str(aug_path),
                "system": SYS_VISION_FULL,
                "user": state_user,
                "assistant": json.dumps({"view_obs": v_obs_aug, "state": cur_state_aug}, ensure_ascii=False),
            })
            counts["map_state_aug"] += 1

            # 4c) Colour-augmented Map-then-Reason BFS planning with generalized cyclic CoT
            n_aug_plans = args.aug_plans_per_frame + (1 if o_info["is_top_up"] else 0)
            for p_i in range(n_aug_plans):
                L = rng.choice([1, 2, 2, 3, 3, 4])
                cands = [int(x) for x in np.nonzero(DIST_TABLE[sidx] == L)[0] if (sidx, int(x)) not in excluded_pairs]
                if not cands:
                    cands = [int(x) for x in np.nonzero(DIST_TABLE[sidx] > 0)[0] if (sidx, int(x)) not in excluded_pairs]
                if not cands:
                    continue
                dst_idx = rng.choice(cands)
                dst_o, dst_d = index_to_state(dst_idx)
                plan = bfs_plan((oid, det), (dst_o, dst_d))
                act = plan[0]
                act_json = format_action_json_cmap(act, cmap)
                thought = build_thought_v2((oid, det), (dst_o, dst_d), act, cmap=cmap)
                dst_state_aug = {
                    "sky_face": cmap[ORIENTATIONS[dst_o]["up_face"]],
                    "front_face": cmap[ORIENTATIONS[dst_o]["front_face"]],
                    strip_key: strip_on_anchor(dst_d, cmap),
                }
                if rng.random() < 0.60:
                    plan_user = (
                        f"This cube uses a novel side-colour palette: {palette_str} (top is white, bottom is magenta).\n"
                        f"Target State: {json.dumps(dst_state_aug)}\n"
                        f"Observe the 3-camera montage, output `view_obs`, `state` (with `sky_face`, `front_face`, `{strip_key}`), "
                        "`thought`, and the next optimal `action` in JSON."
                    )
                else:
                    plan_user = (
                        f"Target State: {json.dumps(dst_state_aug)}\n"
                        f"Observe the 3-camera montage, output `view_obs`, `state` (with `sky_face`, `front_face`, `{strip_key}`), "
                        "`thought`, and the next optimal `action` in JSON."
                    )
                rows_out.append({
                    "id": f"v2_plan_aug_{idx:05d}_{p_i}",
                    "task_type": f"plan_aug_{act[0]}_L{L}",
                    "image": str(aug_path),
                    "system": SYS_VISION_FULL,
                    "user": plan_user,
                    "assistant": json.dumps(
                        {"view_obs": v_obs_aug, "state": cur_state_aug, "thought": thought, "action": act_json},
                        ensure_ascii=False,
                    ),
                })
                counts[f"plan_aug_{act[0]}"] += 1

        # 5) Map-then-Reason BFS planning samples with EXPLICIT cyclic-order CoT (boost turn(+90/-90) coverage!)
        n_plans = args.plans_per_frame + (1 if o_info["is_top_up"] else 0)
        for p_i in range(n_plans):
            L = rng.choice([1, 2, 2, 3, 3, 4])
            cands = [int(x) for x in np.nonzero(DIST_TABLE[sidx] == L)[0] if (sidx, int(x)) not in excluded_pairs]
            if not cands:
                cands = [int(x) for x in np.nonzero(DIST_TABLE[sidx] > 0)[0] if (sidx, int(x)) not in excluded_pairs]
            if not cands:
                continue
            dst_idx = rng.choice(cands)
            dst_o, dst_d = index_to_state(dst_idx)
            plan = bfs_plan((oid, det), (dst_o, dst_d))
            act = plan[0]
            act_json = format_action_json(act)
            thought = build_thought_v2((oid, det), (dst_o, dst_d), act)
            dst_state = {
                "sky_face": ORIENTATIONS[dst_o]["up_colour"],
                "front_face": ORIENTATIONS[dst_o]["front_colour"],
                "on_red_panel_strip": on_red_strip(dst_d),
            }
            rows_out.append({
                "id": f"v2_plan_{idx:05d}_{p_i}",
                "task_type": f"plan_{act[0]}_L{L}",
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
            counts[f"plan_{act[0]}"] += 1

    rng.shuffle(rows_out)
    (out / "sft_c1_v2.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows_out), encoding="utf-8"
    )
    manifest = {
        "selected_frames": len(selected),
        "total_samples": len(rows_out),
        "excluded_test_episodes": len(excluded_eps),
        "excluded_test_pairs": len(excluded_pairs),
        "task_counts": dict(sorted(counts.items())),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("build_c1_v2_dataset_status=PASS " + json.dumps(manifest))


if __name__ == "__main__":
    main()
