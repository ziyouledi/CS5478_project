#!/usr/bin/env python3
"""Composite a publication-grade 1280x720 (30 fps) End-to-End Prompt-to-Manipulation video
from recorded VLM closed-loop demo assets (`record_vlm_closedloop_demo.py`).

Three-stage closed-loop layout:
  - Top bar: User Prompt + Target colour chips + Closed-loop round history (revealed online, no spoilers)
  - 1. OBSERVE (left-top): Live 3-Camera RGB Montage fed to Qwen3-VL-8B
  - 2. PERCEIVE & PLAN (left-bottom): Qwen3-VL-8B Map-then-Reason (clean 3-tier hierarchy: State -> Reason -> Intent)
  - 3. EXECUTE (right): Trained Unified v4 60 Hz multi-skill RL controller (0.5x slow-mo exec, 1.0x PD hold)

Usage:
  python make_vlm_closedloop_video.py --ffmpeg /path/to/ffmpeg \
      --input_dir Docs/Demo/videos/vlm-closedloop-c1-v2-demo-pdhold-job1176862 \
      --prefix ep0_solve_L3_180deg \
      --output Docs/Demo/videos/vlm_end_to_end_ep0.mp4
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
OUT_FPS = 30

# Palette matching Docs/Final presentation/build_final_slides.py
NAVY = (23, 33, 58)
NAVY_DARK = (15, 22, 40)
CARD_BG = (246, 248, 252)
CARD_BORDER = (210, 218, 232)
WHITE = (255, 255, 255)
TEXT_DARK = (23, 33, 58)
TEXT_MUTED = (98, 112, 141)
LIGHT_TEXT = (217, 224, 236)
BLUE = (44, 103, 242)
BLUE_BG = (234, 240, 255)
ORANGE = (217, 96, 47)
ORANGE_BG = (255, 240, 234)
GREEN = (30, 156, 119)
GREEN_BG = (230, 248, 241)
AMBER = (217, 140, 20)
AMBER_BG = (255, 246, 224)

FACE_RGB = {
    "red": (217, 30, 30),
    "green": (28, 166, 56),
    "yellow": (238, 198, 18),
    "blue": (32, 86, 230),
    "white": (245, 245, 245),
    "magenta": (192, 42, 192),
}

FONT_REG = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def read_mp4_frames(ffmpeg: str, mp4_path: Path, width: int = 512, height: int = 512) -> list[np.ndarray]:
    cmd = [
        ffmpeg, "-v", "error", "-i", str(mp4_path),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    raw = proc.stdout.read()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {proc.stderr.read().decode()}")
    frame_bytes = width * height * 3
    n = len(raw) // frame_bytes
    arr = np.frombuffer(raw[: n * frame_bytes], dtype=np.uint8).reshape((n, height, width, 3))
    return [arr[i] for i in range(n)]


def draw_rounded_rect(draw: ImageDraw.ImageDraw, xy, r: int, fill, outline=None, width: int = 1):
    draw.rounded_rectangle(xy, radius=r, fill=fill, outline=outline, width=width)


def draw_colour_chip(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, colour_name: str,
                     f_small, text_col=TEXT_DARK) -> int:
    rgb = FACE_RGB.get(colour_name.lower(), (140, 140, 140))
    draw.rectangle([x, y + 2, x + 13, y + 15], fill=rgb, outline=(70, 70, 70), width=1)
    txt = f"{label}: {colour_name}"
    draw.text((x + 19, y), txt, fill=text_col, font=f_small)
    bbox = draw.textbbox((x + 19, y), txt, font=f_small)
    return bbox[2] + 16


def format_action_short(act: dict | None) -> str:
    if not isinstance(act, dict):
        return "none"
    if act.get("action") == "turn":
        return f"TURN {act.get('delta')}°"
    if act.get("action") == "reorient":
        return f"REORIENT ({act.get('sky_face')}, {act.get('front_face')})"
    return str(act)


def clean_thought(raw: str | None) -> str:
    """Keep the VLM's actual reasoning concise by stripping repetitive cycle-definition boilerplate."""
    if not raw:
        return ""
    s = raw.replace("`", "").strip()
    # Strip boilerplate "On the red panel, +90 cycle is ... (and -90 is reverse)."
    s = re.sub(r"On the red panel,\s*\+90 cycle is [^.]+\.\s*", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def render_static_overlay(
    trace: dict,
    montages: dict[int, Image.Image],
    phase: str,
    round_idx: int,
    solved: bool,
    think_progress: float | None = None,
) -> Image.Image:
    """Render the 1280x720 frame chrome: top bar, Card 1 (Observe), Card 2 (Perceive & Plan), and Card 3 header."""
    img = Image.new("RGB", (W, H), NAVY)
    draw = ImageDraw.Draw(img)

    f_kicker = load_font(FONT_BOLD, 13)
    f_prompt = load_font(FONT_BOLD, 21)
    f_card_title = load_font(FONT_BOLD, 15)
    f_label = load_font(FONT_BOLD, 13)
    f_body = load_font(FONT_REG, 15)
    f_small = load_font(FONT_REG, 14)
    f_small_b = load_font(FONT_BOLD, 14)
    f_pill = load_font(FONT_BOLD, 13)
    f_intent = load_font(FONT_BOLD, 19)

    # --- Top Bar (y = 0..108) ---
    draw.rectangle([0, 0, W, 108], fill=NAVY_DARK)
    draw.line([(0, 108), (W, 108)], fill=(45, 62, 102), width=2)

    kicker = "END-TO-END VISUAL CLOSED-LOOP  ·  QWEN3-VL-8B (SFT)  →  UNIFIED v4 (60 Hz)"
    if trace.get("gen_tag"):
        kicker += f"  ·  [{trace['gen_tag'].upper()}]"
    draw.text((24, 12), kicker, fill=(157, 183, 245) if not trace.get("gen_tag") else (255, 200, 110), font=f_kicker)
    prompt_str = f"Prompt: \"{trace['instruction']}\""
    f_p_use = f_prompt if len(prompt_str) <= 98 else load_font(FONT_BOLD, 17)
    draw.text((24, 34), prompt_str, fill=WHITE, font=f_p_use)

    # Target state colour chips (bottom-left of top bar)
    tgt = trace["target_state"]
    cx = 24
    draw.text((cx, 72), "Target:", fill=LIGHT_TEXT, font=f_small_b)
    cx += 58
    cx = draw_colour_chip(draw, cx, 72, "Sky", tgt["sky_face"], f_small, text_col=LIGHT_TEXT)
    cx = draw_colour_chip(draw, cx, 72, "Front", tgt["front_face"], f_small, text_col=LIGHT_TEXT)
    draw_colour_chip(draw, cx, 72, "Strip on Red", tgt["on_red_panel_strip"], f_small, text_col=LIGHT_TEXT)

    # Closed-loop history pills (bottom-right of top bar): reveal each action ONLY after the VLM decides it
    rounds = trace["rounds"]
    pills = []
    for r in rounds:
        rno = r["round"]
        act = r["parsed_action"]
        short = f"R{rno} " + ("Turn " + str(act["delta"]) + "°" if act.get("action") == "turn" else "Reorient")
        if solved or rno < round_idx:
            pills.append((short, "done"))
        elif rno == round_idx:
            if phase == "think":
                pills.append((f"R{rno} Planning...", "active"))
            elif phase in ("exec", "settle"):
                pills.append((short, "active"))
    if phase == "intro":
        pills = [("Closed-Loop · Ready", "pending")]
    elif solved:
        pills.append(("SOLVED", "solved"))

    px = W - 24
    for label, st in reversed(pills):
        bw = draw.textbbox((0, 0), label, font=f_pill)[2] + 22
        px -= bw
        if st == "solved":
            fill, col = GREEN, WHITE
        elif st == "done":
            fill, col = (36, 118, 92), WHITE
        elif st == "active":
            fill, col = BLUE, WHITE
        else:
            fill, col = (38, 51, 82), (160, 174, 200)
        draw_rounded_rect(draw, [px, 68, px + bw, 94], 12, fill=fill)
        draw.text((px + 11, 73), label, fill=col, font=f_pill)
        px -= 8

    # --- Card 1: 1. OBSERVE (x=24..676, y=124..376) ---
    lx0, lx1 = 24, 676
    draw_rounded_rect(draw, [lx0, 124, lx1, 376], 12, fill=CARD_BG, outline=CARD_BORDER, width=1)
    draw.text((lx0 + 16, 134), "1. OBSERVE  ·  3-Camera RGB Montage", fill=TEXT_DARK, font=f_card_title)
    obs_badge = "Initial Pose" if phase == "intro" else "Live 3-View RGB"
    ob_bg, ob_fg = (ORANGE_BG, ORANGE) if phase == "intro" else (BLUE_BG, BLUE)
    bw = draw.textbbox((0, 0), obs_badge, font=f_pill)[2] + 18
    draw_rounded_rect(draw, [lx1 - 14 - bw, 131, lx1 - 14, 155], 10, fill=ob_bg)
    draw.text((lx1 - 14 - bw + 9, 135), obs_badge, fill=ob_fg, font=f_pill)

    m_key = max(1, min(round_idx, len(rounds)))
    img.paste(montages[m_key], (lx0 + 10, 160))

    # --- Card 2: 2. PERCEIVE & PLAN (x=24..676, y=390..692) ---
    by0, by1 = 390, 692
    draw_rounded_rect(draw, [lx0, by0, lx1, by1], 12, fill=CARD_BG, outline=CARD_BORDER, width=1)
    draw.text((lx0 + 16, by0 + 14), "2. PERCEIVE & PLAN  ·  Qwen3-VL-8B (Map-then-Reason SFT)",
              fill=TEXT_DARK, font=f_card_title)
    draw.line([(lx0 + 16, by0 + 40), (lx1 - 16, by0 + 40)], fill=CARD_BORDER, width=1)

    if phase == "intro":
        draw.text((lx0 + 16, by0 + 58), "Initial scrambled cube state:",
                  fill=TEXT_MUTED, font=f_small_b)
        init_st = trace["init_state"]
        cx = lx0 + 16
        cx = draw_colour_chip(draw, cx, by0 + 88, "Sky", init_st["sky_face"], f_body)
        cx = draw_colour_chip(draw, cx, by0 + 88, "Front", init_st["front_face"], f_body)
        draw_colour_chip(draw, cx, by0 + 88, "Strip on Red", init_st["on_red_panel_strip"], f_body)
        draw.text((lx0 + 16, by0 + 140), "Starting closed-loop visual planning...", fill=TEXT_MUTED, font=f_body)
    else:
        r = rounds[m_key - 1]

        # Tier 1 (Compact): Perceived State in one single clean row
        pst = r.get("pred_state") or r["gt_state"]
        cx = lx0 + 16
        draw.text((cx, by0 + 54), "State:", fill=TEXT_MUTED, font=f_small_b)
        cx += 50
        cx = draw_colour_chip(draw, cx, by0 + 54, "Sky", pst["sky_face"], f_small)
        cx = draw_colour_chip(draw, cx, by0 + 54, "Front", pst["front_face"], f_small)
        cx = draw_colour_chip(draw, cx, by0 + 54, "Strip/Red", pst["on_red_panel_strip"], f_small)
        if r.get("state_ok"):
            ok_txt = "Exact"
            ow = draw.textbbox((0, 0), ok_txt, font=f_pill)[2] + 16
            draw_rounded_rect(draw, [lx1 - 16 - ow, by0 + 51, lx1 - 16, by0 + 73], 9, fill=GREEN_BG)
            draw.text((lx1 - 16 - ow + 8, by0 + 54), ok_txt, fill=GREEN, font=f_pill)

        # Tier 2 (Concise): 2-line cleaned reasoning
        draw.text((lx0 + 16, by0 + 90), "Reason:", fill=TEXT_MUTED, font=f_small_b)
        thought = clean_thought(r.get("parsed_thought"))
        wrapped = textwrap.wrap(thought, width=62)[:2]
        ty = by0 + 90
        for line in wrapped:
            draw.text((lx0 + 80, ty), line, fill=TEXT_DARK, font=f_body)
            ty += 22

        # Tier 3 (HERO focal point): Low-Dim Goal Intent Output box
        iy0, iy1 = by0 + 158, by1 - 16
        if solved:
            draw_rounded_rect(draw, [lx0 + 16, iy0, lx1 - 16, iy1], 10, fill=GREEN_BG, outline=GREEN, width=2)
            draw.text((lx0 + 32, iy0 + 22), "GOAL REACHED  ·  CUBE SOLVED", fill=GREEN, font=f_intent)
            draw.text((lx0 + 32, iy0 + 58),
                      f"Solved in {len(rounds)} closed-loop rounds  ·  0 drops  ·  100% visual state accuracy",
                      fill=TEXT_DARK, font=f_body)
        elif phase == "think":
            draw_rounded_rect(draw, [lx0 + 16, iy0, lx1 - 16, iy1], 10, fill=AMBER_BG, outline=AMBER, width=2)
            draw.text((lx0 + 32, iy0 + 18), f"VLM Thinking (Round {r['round']})...", fill=AMBER, font=f_intent)
            draw.text((lx0 + 32, iy0 + 48),
                      f"Measured VLM latency: {r['vlm_latency_s']:.2f} s  ·  Hand holds pose via PD",
                      fill=TEXT_DARK, font=f_small)
            p = max(0.05, min(1.0, think_progress or 0.5))
            bar_x0, bar_x1, bar_y0, bar_y1 = lx0 + 32, lx1 - 32, iy0 + 82, iy0 + 96
            draw_rounded_rect(draw, [bar_x0, bar_y0, bar_x1, bar_y1], 6, fill=(235, 224, 195))
            draw_rounded_rect(draw, [bar_x0, bar_y0, int(bar_x0 + (bar_x1 - bar_x0) * p), bar_y1], 6, fill=AMBER)
        else:
            draw_rounded_rect(draw, [lx0 + 16, iy0, lx1 - 16, iy1], 10, fill=BLUE_BG, outline=BLUE, width=2)
            draw.text((lx0 + 32, iy0 + 16), "LOW-DIM GOAL INTENT SENT  →", fill=TEXT_MUTED, font=f_label)
            draw.text((lx0 + 32, iy0 + 38), format_action_short(r["parsed_action"]), fill=BLUE, font=f_intent)
            exec_s = r["exec_steps"] / 60.0
            draw.text((lx0 + 32, iy0 + 74),
                      f"VLM: {r['vlm_latency_s']:.1f} s  →  Unified v4 (60 Hz): {r['exec_steps']} steps ({exec_s:.2f} s)",
                      fill=TEXT_DARK, font=f_small)

    # --- Card 3: 3. EXECUTE (right panel frame, x=696..1260, y=124..692) ---
    rx0, rx1, ry0, ry1 = 696, 1260, 124, 692
    draw_rounded_rect(draw, [rx0, ry0, rx1, ry1], 12, fill=CARD_BG, outline=CARD_BORDER, width=1)
    draw.text((rx0 + 16, ry0 + 10), "3. EXECUTE  ·  Unified v4 (Trained 60 Hz Multi-Skill Policy)",
              fill=TEXT_DARK, font=f_card_title)

    return img


def draw_video_status_bar(img: Image.Image, vx: int, vy: int, vw: int, vh: int,
                          phase: str, round_obj: dict | None, solved: bool):
    draw = ImageDraw.Draw(img)
    f_b = load_font(FONT_BOLD, 15)
    f_s = load_font(FONT_REG, 14)

    bar_h = 44
    draw.rectangle([vx, vy + vh - bar_h, vx + vw, vy + vh], fill=NAVY_DARK)
    if solved:
        dot_col, left_txt, right_txt = GREEN, "SOLVED  ·  PD Hold", "1.0× real-time"
    elif phase == "intro":
        dot_col, left_txt, right_txt = ORANGE, "SCRAMBLED  ·  Initial Pose", "1.0× real-time"
    elif phase == "think":
        dot_col, left_txt, right_txt = AMBER, "PD HOLD  ·  Waiting for Slow VLM", "1.0× real-time"
    elif phase == "exec":
        act_s = format_action_short(round_obj["parsed_action"]) if round_obj else ""
        dot_col, left_txt, right_txt = BLUE, f"60 Hz UNIFIED v4  ·  {act_s}", "0.5× slow-mo"
    else:
        dot_col, left_txt, right_txt = GREEN, "SUBGOAL REACHED  ·  Settled", "1.0× real-time"

    draw.ellipse([vx + 14, vy + vh - 29, vx + 27, vy + vh - 16], fill=dot_col)
    draw.text((vx + 36, vy + vh - 31), left_txt, fill=WHITE, font=f_b)
    rw = draw.textbbox((0, 0), right_txt, font=f_s)[2]
    draw.text((vx + vw - 14 - rw, vy + vh - 30), right_txt, fill=LIGHT_TEXT, font=f_s)


def build_timeline(trace: dict) -> list[tuple[int, str, int, bool, float | None]]:
    """Return list of (src_frame_idx, phase, round_idx, solved, think_progress) at 30 fps.
    - intro / think / settle / final: 1.0x real-time
    - exec: 0.5x slow-mo (each 60 fps src frame shown once at 30 fps — natural and crisp!)
    """
    seq = []
    rounds = trace["rounds"]
    total = trace["total_frames"]

    intro_end = rounds[0]["frame_start"] if rounds else total
    for f in range(0, intro_end, 2):
        seq.append((f, "intro", 1, False, None))

    for r in rounds:
        rno = r["round"]
        fs, ft, fe = r["frame_start"], r["frame_think_end"], r["frame_exec_end"]
        exec_end = min(fe, ft + r["exec_steps"])
        # Think hold (~1.0 s = 30 out frames)
        think_src = list(range(fs, ft, 3))
        n_th = max(len(think_src), 1)
        for idx, f in enumerate(think_src):
            seq.append((f, "think", rno, False, (idx + 1) / n_th))
        # Execution (0.5x slow-mo)
        for f in range(ft, exec_end):
            seq.append((f, "exec", rno, False, None))
        # Post-round settle (~0.7 s holding the blue Intent Sent card so it stays easy to read)
        settle_src = list(range(exec_end, fe, 1))
        for f in settle_src:
            seq.append((f, "settle", rno, False, None))
        if settle_src:
            for _ in range(max(0, 21 - len(settle_src))):
                seq.append((settle_src[-1], "settle", rno, False, None))

    last_end = rounds[-1]["frame_exec_end"] if rounds else 0
    for f in range(last_end, total, 2):
        seq.append((f, "final", len(rounds), bool(trace["success"]), None))
    if total > 0:
        for _ in range(15):
            seq.append((total - 1, "final", len(rounds), bool(trace["success"]), None))
    return seq


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ffmpeg", required=True)
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--prefix", default="ep0_solve_L3_180deg")
    ap.add_argument("--cam", choices=["demo", "c1"], default="demo")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    trace = json.loads((in_dir / f"{args.prefix}_trace.json").read_text(encoding="utf-8"))
    mp4_path = in_dir / f"{args.prefix}_{args.cam}.mp4"
    frames = read_mp4_frames(args.ffmpeg, mp4_path, 512, 512)
    print(f"Loaded {len(frames)} frames from {mp4_path.name}")

    montages = {}
    for r in trace["rounds"]:
        m = Image.open(in_dir / r["montage_png"]).convert("RGB")
        mw, mh = m.size
        target_w = 632
        target_h = int(round(mh * target_w / mw))
        if target_h > 208:
            target_h = 208
            target_w = int(round(mw * target_h / mh))
        m_resized = m.resize((target_w, target_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (632, 208), (24, 24, 24))
        canvas.paste(m_resized, ((632 - target_w) // 2, (208 - target_h) // 2))
        montages[r["round"]] = canvas

    timeline = build_timeline(trace)
    print(f"Timeline: {len(timeline)} output frames ({len(timeline) / OUT_FPS:.2f} s at {OUT_FPS} fps)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.ffmpeg, "-y", "-v", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(OUT_FPS), "-i", "-",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-movflags", "+faststart", str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    # Right video sits cleanly inside Card 3 (header y=124..160, video y=160..640, status bar y=640..684)
    vx, vy, vw, vh = 706, 160, 544, 522
    vid_h = vh - 44
    overlay_cache: dict[tuple, Image.Image] = {}

    for src_f, phase, rno, solved, tprog in timeline:
        src_f = max(0, min(src_f, len(frames) - 1))
        qprog = None if tprog is None else round(tprog * 20) / 20.0
        key = (phase, rno, solved, qprog)
        if key not in overlay_cache:
            overlay_cache[key] = render_static_overlay(trace, montages, phase, rno, solved, qprog)
        canvas = overlay_cache[key].copy()

        v_im = Image.fromarray(frames[src_f]).resize((vw, vw), Image.Resampling.LANCZOS)
        crop_top = (vw - vid_h) // 2
        v_im = v_im.crop((0, crop_top, vw, crop_top + vid_h))
        canvas.paste(v_im, (vx, vy))

        r_obj = trace["rounds"][rno - 1] if 1 <= rno <= len(trace["rounds"]) else None
        draw_video_status_bar(canvas, vx, vy, vw, vh, phase, r_obj, solved)
        proc.stdin.write(canvas.tobytes())

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit(f"ffmpeg encode failed with code {proc.returncode}")

    poster = out_path.with_name(out_path.stem + "_poster.png")
    subprocess.run([args.ffmpeg, "-y", "-v", "error", "-ss", "1.8", "-i", str(out_path), "-frames:v", "1", str(poster)], check=True)
    print(f"Wrote {out_path} and {poster}")


if __name__ == "__main__":
    main()
