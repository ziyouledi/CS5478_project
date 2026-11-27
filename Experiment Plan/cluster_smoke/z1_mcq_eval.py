#!/usr/bin/env python3
"""Z1 v2: cube-state perception as a standard multiple-choice benchmark (GOAL_LEVEL_RL_PLAN.md Z1 评测规范 step 2).

Follows the conventions of the public benchmarks instead of a hand-written prompt:
  - Template: VLMEvalKit ImageMCQ (mcq_common.py), the same path that reproduced BLINK / MMStar / EmbSpatial in
    job 1166604. Images first, then Hint / Question / Options / instruction. No per-camera description (benchmarks
    give none). The templates below are frozen (TEMPLATE_VERSION); do not tune them on the test split.
  - Greedy decoding (temperature 0), one completion per prompt.
  - CircularEval (MMBench): a question with n content options is asked n times with the options rotated; it counts
    as correct only if every rotation is answered correctly. "Cannot be determined" (DORI) is always the last
    option and is not rotated.
  - Coarse to fine: up (which face points up) -> turned (is the top layer turned) -> band (which colour).
  - Answerability from segmentation pixel counts: a (question, camera set) is answerable if the relevant faces
    have enough visible pixels in at least one of the cameras. Accuracy is over answerable items; for the rest the
    rate of choosing "Cannot be determined" is reported.
  - Baselines: uniform random (single pass and under CircularEval) and the best constant answer (random++ in
    3DSRBench: always the same content, which CircularEval does not penalise).
  - dev / test split by episode (z1_build_mcq.py); test is the reported number.
"""

from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

from mcq_common import LETTERS, build_prompt, extract, image_url

FACE_LABELS = [
    "base_px", "base_py", "base_nx", "base_ny", "base_nz",
    "top_px", "top_py", "top_nx", "top_ny", "top_pz",
    "stripe", "core_base", "core_top", "hand",
]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", required=True)
parser.add_argument("--frames", required=True, help="z1_build_mcq.py output")
parser.add_argument("--output_dir", required=True)
parser.add_argument("--conditions", default="c1,top,s040+s220,c1+s040+s220,c1+s040+s220@nohand")
parser.add_argument("--splits", default="dev,test")
parser.add_argument("--max_tokens", type=int, default=4096)
parser.add_argument("--max_model_len", type=int, default=32768)
parser.add_argument("--gpu_mem_gib", type=float, default=0.0)
parser.add_argument("--chunk", type=int, default=500, help="Prompts per llm.chat call (host memory, job 1166173)")
parser.add_argument("--min_face_px", type=int, default=300, help="Visible pixels for a face to count as seen")
parser.add_argument("--min_band_px", type=int, default=100, help="Visible pixels for a top-layer strip")
parser.add_argument("--limit", type=int, default=0, help="First N frames per split (smoke test)")
parser.add_argument("--visual_mode", default="raw", choices=["raw", "crop", "montage", "seam", "som"],
                    help="P1 visual input transform: raw (v2 baseline), crop (ROI zoom to 384x384), "
                         "montage (single composite image of cropped views with headers), "
                         "seam (crop + high-contrast layer split boundary), "
                         "som (crop + Set-of-Mark numeric tags [1]/[2] on base face and top strip)")
parser.add_argument("--crop_size", type=int, default=384, help="Output resolution per view for crop/montage/seam/som")
parser.add_argument("--crop_pad", type=float, default=0.30, help="Padding fraction around the cube segmentation bbox")
parser.add_argument("--question_set", default="v2", choices=["v2", "v3"],
                    help="v2: original frozen up/turned/band; "
                         "v3: disambiguated sky_face / cap_aligned / pointer_base (black bar & red 1/3 strip on 2/3 base)")
parser.add_argument("--ood_color_mode", default="none", choices=["none", "reverse_cycle", "novel_palette"],
                    help="OOD appearance generalization test: `reverse_cycle` swaps green (py) and blue (ny); "
                         "`novel_palette` recolors (red,green,yellow,blue)->(orange,cyan,purple,pink) via segmentation masks.")
args = parser.parse_args()

TEMPLATE_VERSION = "z1-mcq-v2"
CBD = "Cannot be determined"
HINT = ("The images show the same moment from {n} fixed camera{s}. A robot hand, palm up, holds a cube whose faces "
        "are coloured. The hand may hide parts of the cube.")
HINT_MONTAGE = ("The image is a side-by-side panel showing the same moment from {n} fixed camera{s}. A robot hand, "
                "palm up, holds a cube whose faces are coloured. The hand may hide parts of the cube.")
QUESTIONS = {
    "up": ("Which colour covers most of the cube face that points up (toward the sky)?",
           ["white", "magenta", "red", "green", "yellow", "blue"]),
    "turned": ("Apart from the white face and the magenta face, does each face of the cube show a single colour, or "
               "does each show a wide part of one colour and a narrow strip of another colour?",
               ["Each face shows a single colour", "Each face shows a wide part and a narrow strip of another colour"]),
    "band": ("Find the cube face that is mostly red. What colour is the narrow strip along one of its edges?",
             ["There is no strip; the face is red only", "blue", "yellow", "green"]),
}
QUESTIONS_SOM = {
    "up": QUESTIONS["up"],
    "turned": ("Look at the cube face where [1] marks the large 2/3 panel and [2] marks the narrow 1/3 strip along "
               "its split edge. Do [1] and [2] show the same colour, or two different colours?",
               ["[1] and [2] show the same colour", "[1] and [2] show two different colours"]),
    "band": ("Look at the cube face where [1] marks the red 2/3 panel and [2] marks the narrow 1/3 strip along its "
             "split edge. What colour is the strip marked [2]?",
             ["red (same colour as [1]; there is no different strip)", "blue", "yellow", "green"]),
}
QUESTIONS_V3 = {
    "up": ("Which colour covers most of the cube face that points straight up (toward the sky, away from the robot palm)?",
           ["white", "magenta", "red", "green", "yellow", "blue"]),
    "turned": ("The cube consists of a larger 2/3 base and a thinner 1/3 rotatable cap (which has a white face with a "
               "black bar). On the four coloured side faces (red, green, yellow, blue), does each thin 1/3 strip have "
               "the same colour as the large 2/3 panel it sits on, or a different colour?",
               ["Aligned: each thin 1/3 strip matches the colour of its large 2/3 panel",
                "Misaligned: the thin 1/3 strips have different colours from the large 2/3 panels they sit on"]),
    "band": ("The thinner 1/3 white cap has a black bar on its white face pointing toward its own red 1/3 side strip. "
             "Which colour of the large 2/3 base face is that red 1/3 strip (and the black bar) currently aligned with?",
             ["red", "green", "yellow", "blue"]),
}
NOVEL_PALETTE_MAP = {"red": "orange", "green": "cyan", "yellow": "purple", "blue": "pink"}
QUESTIONS_NOVEL = {
    "up": ("Which colour covers most of the cube face that points up (toward the sky)?",
           ["white", "magenta", "orange", "cyan", "purple", "pink"]),
    "turned": QUESTIONS["turned"],
    "band": ("Find the cube face that is mostly orange. What colour is the narrow strip along one of its edges?",
             ["There is no strip; the face is orange only", "pink", "purple", "cyan"]),
}
TRUTH_TO_OPTION_NOVEL = {"turned": {"no": 0, "yes": 1}, "band": {"none": 0, "pink": 1, "purple": 2, "cyan": 3}}
POINTER_COLOURS = ["red", "green", "yellow", "blue"]  # detent 0, 1, 2, 3 -> base_px, base_py, base_nx, base_ny
TRUTH_TO_OPTION = {"turned": {"no": 0, "yes": 1}, "band": {"none": 0, "blue": 1, "yellow": 2, "green": 3}}
SIDES = ["px", "py", "nx", "ny"]
LABEL_INDEX = {name: i + 1 for i, name in enumerate(FACE_LABELS)}
CUBE_IDS = [LABEL_INDEX[name] for name in FACE_LABELS if name != "hand"]
BASE_IDS = [LABEL_INDEX[f"base_{s}"] for s in SIDES] + [LABEL_INDEX["base_nz"], LABEL_INDEX["core_base"]]
TOP_IDS = [LABEL_INDEX[f"top_{s}"] for s in SIDES] + [LABEL_INDEX["top_pz"], LABEL_INDEX["stripe"], LABEL_INDEX["core_top"]]
CAM_TITLES = {"c1": "VIEW 1 (FRONT-TOP)", "s040": "VIEW 2 (LEFT-SIDE)", "s220": "VIEW 3 (RIGHT-SIDE)",
              "s130": "VIEW (BACK-SIDE)", "top": "VIEW (TOP-DOWN)"}


def render_visual(frame: dict, cam: str, nohand: bool, level: str):
    """Return a transformed PIL Image for (frame, cam, nohand, level) under args.visual_mode."""
    import io
    import numpy as np
    from PIL import Image, ImageDraw

    run_dir = Path(frame["run_dir"])
    mode = "nohand" if nohand else "normal"
    img_path = run_dir / "images" / f"s{frame['step']:04d}_e{frame['env']:03d}_{cam}{'_nohand' if nohand else ''}.png"
    img = Image.open(img_path).convert("RGB")
    if args.visual_mode == "raw" and args.ood_color_mode == "none":
        return img

    npz_path = run_dir / "labels" / f"s{frame['step']:04d}.npz"
    with np.load(npz_path) as data:
        seg = data[f"{cam}_{mode}"][frame["env"]]  # (H, W) uint8
        seg_nohand = data[f"{cam}_nohand"][frame["env"]]

    H, W = seg.shape
    if args.ood_color_mode == "reverse_cycle":
        # Swap green (py) and blue (ny) while preserving 3D diffuse shading
        arr_f = np.asarray(img, dtype=np.float32) / 255.0
        out_f = arr_f.copy()
        rgb_g = np.array([0.10, 0.65, 0.20], dtype=np.float32)
        rgb_b = np.array([0.10, 0.30, 0.90], dtype=np.float32)
        for prefix in ("base_", "top_"):
            m_g = seg == LABEL_INDEX[f"{prefix}py"]
            m_b = seg == LABEL_INDEX[f"{prefix}ny"]
            if m_g.any():
                shade_g = (arr_f[m_g].max(axis=-1, keepdims=True) / float(rgb_g.max())).clip(0.15, 1.35)
                out_f[m_g] = (rgb_b * shade_g).clip(0.0, 1.0)
            if m_b.any():
                shade_b = (arr_f[m_b].max(axis=-1, keepdims=True) / float(rgb_b.max())).clip(0.15, 1.35)
                out_f[m_b] = (rgb_g * shade_b).clip(0.0, 1.0)
        img = Image.fromarray((out_f * 255.0).round().astype(np.uint8))
        if args.visual_mode == "raw":
            return img
    elif args.ood_color_mode == "novel_palette":
        # Recolor (red, green, yellow, blue) -> (orange, cyan, purple, pink) preserving 3D diffuse shading
        arr_f = np.asarray(img, dtype=np.float32) / 255.0
        out_f = arr_f.copy()
        recolor_spec = {
            "px": (np.array([0.85, 0.08, 0.08], dtype=np.float32), np.array([0.95, 0.45, 0.05], dtype=np.float32)),
            "py": (np.array([0.10, 0.65, 0.20], dtype=np.float32), np.array([0.05, 0.78, 0.85], dtype=np.float32)),
            "nx": (np.array([0.95, 0.80, 0.05], dtype=np.float32), np.array([0.55, 0.15, 0.85], dtype=np.float32)),
            "ny": (np.array([0.10, 0.30, 0.90], dtype=np.float32), np.array([0.95, 0.35, 0.65], dtype=np.float32)),
        }
        for s_name, (orig_rgb, tgt_rgb) in recolor_spec.items():
            orig_peak = float(orig_rgb.max())
            for prefix in ("base_", "top_"):
                m = seg == LABEL_INDEX[f"{prefix}{s_name}"]
                if m.any():
                    shade = (arr_f[m].max(axis=-1, keepdims=True) / max(orig_peak, 1e-3)).clip(0.15, 1.35)
                    out_f[m] = (tgt_rgb * shade).clip(0.0, 1.0)
        img = Image.fromarray((out_f * 255.0).round().astype(np.uint8))
        if args.visual_mode == "raw":
            return img

    arr = np.array(img, copy=True)

    # P1.3: Seam enhancement (draw high-contrast cyan/white split boundary between base and top layer)
    if args.visual_mode == "seam":
        base_mask = np.isin(seg, BASE_IDS)
        top_mask = np.isin(seg, TOP_IDS)
        # 2-pixel dilation of base and top to find their interface seam
        base_dil = base_mask.copy()
        top_dil = top_mask.copy()
        for _ in range(3):
            base_dil[1:, :] |= base_dil[:-1, :]
            base_dil[:-1, :] |= base_dil[1:, :]
            base_dil[:, 1:] |= base_dil[:, :-1]
            base_dil[:, :-1] |= base_dil[:, 1:]
            top_dil[1:, :] |= top_dil[:-1, :]
            top_dil[:-1, :] |= top_dil[1:, :]
            top_dil[:, 1:] |= top_dil[:, :-1]
            top_dil[:, :-1] |= top_dil[:, 1:]
        seam = base_dil & top_dil & (seg != LABEL_INDEX["hand"])
        arr[seam] = [0, 255, 255]  # bright cyan split line (distinct from all 6 face colours)
        img = Image.fromarray(arr)

    # Compute ROI crop box from nohand cube segmentation mask so hand never shifts the framing
    cube_mask = np.isin(seg_nohand, CUBE_IDS)
    ys, xs = np.nonzero(cube_mask)
    if len(ys) > 0:
        ymin, ymax = int(ys.min()), int(ys.max())
        xmin, xmax = int(xs.min()), int(xs.max())
        cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
        half = max(xmax - xmin + 1, ymax - ymin + 1) * (0.5 + args.crop_pad)
        half = max(half, 64.0)
        x0 = max(0, int(round(cx - half)))
        y0 = max(0, int(round(cy - half)))
        x1 = min(W, int(round(cx + half)))
        y1 = min(H, int(round(cy + half)))
    else:
        x0, y0, x1, y1 = W // 4, H // 4, 3 * W // 4, 3 * H // 4

    cropped = img.crop((x0, y0, x1, y1)).resize((args.crop_size, args.crop_size), Image.Resampling.BICUBIC)

    # P1.2: Set-of-Mark (SoM) numeric tags [1] on base side and [2] on top strip above it
    if args.visual_mode == "som" and level in ("turned", "band"):
        seg_crop = seg[y0:y1, x0:x1]
        ch, cw = seg_crop.shape
        scale_x = args.crop_size / max(cw, 1)
        scale_y = args.crop_size / max(ch, 1)
        candidate_sides = ["px"] if level == "band" else SIDES
        best_side = None
        best_score = -1
        for s in candidate_sides:
            b_id = LABEL_INDEX[f"base_{s}"]
            t_id = LABEL_INDEX[frame["above"][s]]
            nb = int((seg_crop == b_id).sum())
            nt = int((seg_crop == t_id).sum())
            if nb >= 40 and nt >= 15 and min(nb, nt * 3) > best_score:
                best_score = min(nb, nt * 3)
                best_side = (b_id, t_id)
        if best_side is not None:
            draw = ImageDraw.Draw(cropped)
            for tag, lid in (("1", best_side[0]), ("2", best_side[1])):
                ry, rx = np.nonzero(seg_crop == lid)
                if len(ry) > 0:
                    tx = int(round(float(rx.mean()) * scale_x))
                    ty = int(round(float(ry.mean()) * scale_y))
                    # Draw small high-contrast pill tag offset or right at centroid with thin pointer
                    r = 10
                    bx = max(r + 2, min(args.crop_size - r - 2, tx))
                    by = max(r + 2, min(args.crop_size - r - 2, ty))
                    draw.rectangle([bx - r, by - r, bx + r, by + r], fill=(0, 0, 0), outline=(255, 255, 255), width=2)
                    draw.text((bx - 3, by - 6), tag, fill=(255, 255, 255))

    return cropped


def encode_pil(img) -> str:
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return image_url(buf.getvalue())


def build_condition_urls(frame: dict, cams: list[str], nohand: bool, level: str, cache: dict) -> list[str]:
    """Return list of data-URLs for (frame, cams, nohand, level)."""
    from PIL import Image, ImageDraw

    lvl_key = level if (args.visual_mode == "som" and level in ("turned", "band")) else "all"
    key = (frame["id"], tuple(cams), nohand, args.visual_mode, args.ood_color_mode, lvl_key)
    if key in cache:
        return cache[key]

    if args.visual_mode == "raw" and args.ood_color_mode == "none":
        run_dir = Path(frame["run_dir"])
        urls = []
        for c in cams:
            p = run_dir / "images" / f"s{frame['step']:04d}_e{frame['env']:03d}_{c}{'_nohand' if nohand else ''}.png"
            pkey = str(p)
            if pkey not in cache:
                cache[pkey] = image_url(p.read_bytes())
            urls.append(cache[pkey])
        cache[key] = urls
        return urls

    views = [render_visual(frame, c, nohand, level) for c in cams]
    if args.visual_mode == "montage" and len(views) > 1:
        header_h = 28
        pad = 6
        S = args.crop_size
        total_w = len(views) * S + (len(views) - 1) * pad
        total_h = S + header_h
        canvas = Image.new("RGB", (total_w, total_h), (24, 24, 24))
        draw = ImageDraw.Draw(canvas)
        for idx, (c, v) in enumerate(zip(cams, views)):
            x_off = idx * (S + pad)
            title = CAM_TITLES.get(c, c.upper())
            draw.text((x_off + 10, 7), title, fill=(255, 255, 255))
            canvas.paste(v, (x_off, header_h))
        urls = [encode_pil(canvas)]
    else:
        urls = [encode_pil(v) for v in views]

    cache[key] = urls
    return urls


def active_questions() -> dict[str, tuple[str, list[str]]]:
    if args.ood_color_mode == "novel_palette":
        return QUESTIONS_NOVEL
    if args.visual_mode == "som":
        return QUESTIONS_SOM
    if args.question_set == "v3":
        return QUESTIONS_V3
    return QUESTIONS


def truth_value(level: str, frame: dict) -> str:
    if args.question_set == "v3" and level == "band":
        val = POINTER_COLOURS[int(frame["detent"]) % 4]
    else:
        val = frame["truth"][level]
    if args.ood_color_mode == "reverse_cycle" and level in ("up", "band"):
        if val == "green":
            return "blue"
        if val == "blue":
            return "green"
    if args.ood_color_mode == "novel_palette" and level in ("up", "band"):
        return NOVEL_PALETTE_MAP.get(val, val)
    return val


def truth_index(level: str, frame: dict) -> int:
    value = truth_value(level, frame)
    q_map = active_questions()
    if level == "up" or (args.question_set == "v3" and level == "band"):
        return q_map[level][1].index(value)
    if args.ood_color_mode == "novel_palette":
        return TRUTH_TO_OPTION_NOVEL[level][value]
    return TRUTH_TO_OPTION[level][value]


def answerable(level: str, frame: dict, cams: list[str], mode: str) -> bool:
    pix = [frame["pixels"][c][mode] for c in cams]
    if level == "up":
        face = {"pz": "top_pz", "nz": "base_nz"}.get(frame["up_face"], f"base_{frame['up_face']}")
        return any(p[face] >= args.min_face_px for p in pix)
    if args.question_set == "v3" and level == "band":
        # Answerable if either (a) the cap's red 1/3 strip (top_px) and the 2/3 base panel it sits on are visible,
        # OR (b) the white face (top_pz) + black bar (stripe) are visible in one view and the target base panel is visible.
        target_base = f"base_{SIDES[int(frame['detent']) % 4]}"
        side_cue = any(p.get(target_base, 0) >= args.min_face_px and p.get("top_px", 0) >= args.min_band_px for p in pix)
        bar_cue = any(p.get("top_pz", 0) >= args.min_face_px and p.get(target_base, 0) >= args.min_face_px for p in pix)
        return side_cue or bar_cue
    sides = SIDES if level == "turned" else ["px"]
    return any(p[f"base_{s}"] >= args.min_face_px and p[frame["above"][s]] >= args.min_band_px
               for p in pix for s in sides)


def rotations(level: str) -> list[list[str]]:
    opts = active_questions()[level][1]
    return [opts[s:] + opts[:s] for s in range(len(opts))]


def main() -> None:
    import torch
    from vllm import LLM, SamplingParams

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = [json.loads(line) for line in open(args.frames, encoding="utf-8")]
    splits = args.splits.split(",")
    by_split = {s: [f for f in frames if f["split"] == s] for s in splits}
    if args.limit:
        by_split = {s: v[: args.limit] for s, v in by_split.items()}
    conditions = []
    for spec in args.conditions.split(","):
        cams_spec, _, flag = spec.partition("@")
        conditions.append((spec, cams_spec.split("+"), flag == "nohand"))

    q_map = active_questions()
    free, total = torch.cuda.mem_get_info()
    utilization = 0.9 if args.gpu_mem_gib <= 0 else args.gpu_mem_gib * 2**30 / total
    print(f"gpu_free_gib={free / 2**30:.1f} vllm_utilization={utilization:.3f} visual_mode={args.visual_mode}",
          flush=True)
    max_imgs = 1 if args.visual_mode == "montage" else max(len(c[1]) for c in conditions)
    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              limit_mm_per_prompt={"image": max_imgs},
              gpu_memory_utilization=utilization, seed=3407)
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    summary = {"model": args.model, "frames": args.frames,
               "template_version": f"{TEMPLATE_VERSION}-{args.question_set}",
               "question_set": args.question_set,
               "visual_mode": args.visual_mode, "crop_size": args.crop_size, "crop_pad": args.crop_pad,
               "hint": HINT_MONTAGE if args.visual_mode == "montage" else HINT,
               "questions": q_map, "cbd": CBD, "decoding": "greedy", "max_tokens": args.max_tokens,
               "min_face_px": args.min_face_px, "min_band_px": args.min_band_px, "results": {}}
    responses = open(out / "responses.jsonl", "w", encoding="utf-8")
    image_cache: dict = {}

    for split, split_frames in by_split.items():
        for spec, cams, nohand in conditions:
            t0 = time.time()
            mode = "nohand" if nohand else "normal"
            hint_tpl = HINT_MONTAGE if (args.visual_mode == "montage" and len(cams) > 1) else HINT
            hint = hint_tpl.format(n=len(cams), s="s" if len(cams) > 1 else "")
            jobs = []  # (frame, level, rotation index, options dict, truth letter, messages)
            for frame in split_frames:
                for level, (question, opts) in q_map.items():
                    urls = build_condition_urls(frame, cams, nohand, level, image_cache)
                    truth = opts[truth_index(level, frame)]
                    for r, rotated in enumerate(rotations(level)):
                        options = dict(zip(LETTERS, rotated + [CBD]))
                        letter = LETTERS[rotated.index(truth)]
                        content = [{"type": "image_url", "image_url": {"url": u}} for u in urls]
                        content.append({"type": "text", "text": build_prompt(question, options, hint)})
                        jobs.append((frame, level, r, options, letter, [{"role": "user", "content": content}]))
            outputs = []
            for start in range(0, len(jobs), args.chunk):
                outputs += llm.chat([j[5] for j in jobs[start:start + args.chunk]], params, use_tqdm=False)
            image_cache.clear()

            # (frame id, level) -> list of (ok, chose_cbd)
            passes = collections.defaultdict(list)
            tally = collections.Counter()
            for (frame, level, r, options, letter, _), result in zip(jobs, outputs):
                completion = result.outputs[0]
                pred = extract(completion.text, options)
                ok = pred == letter
                cbd = pred is not None and options[pred] == CBD
                passes[(frame["id"], level)].append((ok, cbd))
                tally[f"{level}/prompts"] += 1
                tally[f"{level}/unmatched"] += pred is None
                tally[f"{level}/truncated"] += completion.finish_reason == "length"
                tally[f"{level}/tokens"] += len(completion.token_ids)
                responses.write(json.dumps({
                    "split": split, "condition": spec, "id": frame["id"], "level": level, "rotation": r,
                    "truth": letter, "pred": pred, "pred_text": options.get(pred), "ok": ok,
                    "finish": completion.finish_reason, "text": completion.text}) + "\n")

            record = {"n_frames": len(split_frames), "seconds": time.time() - t0, "levels": {}}
            frame_by_id = {f["id"]: f for f in split_frames}
            all_ok = collections.defaultdict(lambda: True)
            all_answerable = collections.defaultdict(lambda: True)
            for level, (_, opts) in q_map.items():
                n_opts = len(opts) + 1
                stats = collections.Counter()
                strata = collections.defaultdict(lambda: [0, 0])
                truths = collections.Counter()
                for fid, frame in frame_by_id.items():
                    p = passes[(fid, level)]
                    circ, vanilla = all(ok for ok, _ in p), p[0][0]
                    ans = answerable(level, frame, cams, mode)
                    all_answerable[fid] &= ans
                    all_ok[fid] &= circ
                    if ans:
                        tv = truth_value(level, frame)
                        truths[tv] += 1
                        stats["n"] += 1
                        stats["circular"] += circ
                        stats["vanilla"] += vanilla
                        stats["cbd_any"] += any(c for _, c in p)
                        for key in (f"truth={tv}", f"detent={frame['detent']}",
                                    f"up={frame['truth']['up']}"):
                            strata[key][0] += circ
                            strata[key][1] += 1
                    else:
                        stats["n_unanswerable"] += 1
                        stats["cbd_unanswerable"] += any(c for _, c in p)
                n = max(stats["n"], 1)
                prompts = max(tally[f"{level}/prompts"], 1)
                record["levels"][level] = {
                    "n_answerable": stats["n"],
                    "acc_circular": stats["circular"] / n,
                    "acc_vanilla": stats["vanilla"] / n,
                    "random_vanilla": 1 / n_opts,
                    "random_circular": (1 / n_opts) ** (n_opts - 1),
                    "best_constant": max(truths.values()) / n if truths else 0.0,
                    "best_constant_answer": truths.most_common(1)[0][0] if truths else None,
                    "cbd_rate_answerable": stats["cbd_any"] / n,
                    "n_unanswerable": stats["n_unanswerable"],
                    "cbd_rate_unanswerable": stats["cbd_unanswerable"] / max(stats["n_unanswerable"], 1),
                    "unmatched": tally[f"{level}/unmatched"] / prompts,
                    "truncated": tally[f"{level}/truncated"] / prompts,
                    "mean_tokens": tally[f"{level}/tokens"] / prompts,
                    "strata_circular": {k: [a / b, b] for k, (a, b) in sorted(strata.items())},
                }
            full = [fid for fid in frame_by_id if all_answerable[fid]]
            record["all_levels_circular"] = sum(all_ok[f] for f in full) / max(len(full), 1)
            record["n_all_answerable"] = len(full)
            summary["results"][f"{split}/{spec}"] = record
            line = {lv: {k: round(v, 3) for k, v in rec.items() if k in ("acc_circular", "acc_vanilla",
                                                                          "best_constant", "unmatched")}
                    for lv, rec in record["levels"].items()}
            print(f"z1_mcq={split}/{spec} all={record['all_levels_circular']:.3f} n_all={len(full)} "
                  + json.dumps(line), flush=True)
            (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    responses.close()
    print("z1_mcq_eval_status=PASS")


if __name__ == "__main__":
    main()
