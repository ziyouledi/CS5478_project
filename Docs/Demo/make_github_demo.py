#!/usr/bin/env python3
"""Assemble:
  1. GitHub README Teaser (`Docs/Demo/videos/github_teaser.mp4`):
     4 end-to-end visual closed-loop VLM Rubik's-cube solving conditions with crisp divider cards:
       - Standard (ep0, L=3)
       - Zero-Shot OOD Lighting: Warm Dim (gen0, L=3)
       - Zero-Shot OOD Lighting: Cool Bright (gen1, L=2)
       - Zero-Shot Unseen Conversational Prompt (gen2)
  2. Full GitHub End-to-End Demo Video (`Docs/Demo/videos/github_full_demo.mp4`, ~2 min, 1280x720, 30 fps):
     - Act 0: Opening Title & Core Question
     - Act 1: Task 1 (SO(3)) — 60 Hz Content, 2 Hz Rate Reversal, 0.1 s Latency Tolerance
     - Act 2: Staleness Mechanism (widescreen) & Task 2 (Z-Axis) Stale Goal Trap
     - Act 3: Task 3 — Single-Hand 6-Colour Rubik's Cube (60 Hz Multi-Skill + E3 10 Hz Rate Reversal)
     - Act 4: End-to-End Visual VLM Closed-Loop Control & Zero-Shot Generalization (Plan C)
     - Act 5: Closing Takeaway Card

Usage:
  python make_github_demo.py --ffmpeg /path/to/ffmpeg [--skip_vlm]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
FINAL_PRES = REPO / "Docs" / "Final presentation"
CLIPS_DIR = FINAL_PRES / "clips"
VLM_STD_DIR = REPO / "Docs" / "Demo" / "videos" / "vlm-closedloop-c1-v2-demo-pdhold-job1176862"
VLM_GEN_DIR = REPO / "Docs" / "Demo" / "videos" / "vlm-closedloop-c1-v2-gen-demo-job1177273"
VLM_SUITE_DIR = REPO / "Docs" / "Demo" / "videos" / "vlm-gen-suite-job1177677" / "videos"
TASK3_DIR = REPO / "Docs" / "Demo" / "videos" / "task3-unified-rubik-demo-job1177271"
OUT_DIR = REPO / "Docs" / "Demo" / "videos"

W, H, FPS = 1280, 720, 30
DEMO_JOBS = ("1125119", "1125120", "1125391", "1127897", "1127918", "1129142")

FONT_REG = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

NAVY = (23, 33, 58)
NAVY_DARK = (15, 22, 40)
WHITE = (255, 255, 255)
LIGHT_TEXT = (217, 224, 236)
KICKER_BLUE = (157, 183, 245)
BLUE = (44, 103, 242)
ORANGE = (217, 96, 47)
GREEN = (30, 156, 119)
AMBER = (230, 160, 40)

CROP_W, CROP_H, CROP_X = 760, 720, 330
TILE_H = 584
TILE_W = 2 * round(CROP_W * TILE_H / CROP_H / 2)
TOP = 104


def load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise SystemExit(f"Command failed:\n{' '.join(cmd)}\n{r.stderr[-2500:]}")


def esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace(":", "\\:").replace("'", "’").replace(",", "\\,")


def load_clip(run_name: str) -> dict:
    hits = [CLIPS_DIR / f"{run_name}-job{j}_0" for j in DEMO_JOBS if (CLIPS_DIR / f"{run_name}-job{j}_0").is_dir()]
    if len(hits) != 1:
        raise SystemExit(f"Expected 1 clip for {run_name}, found {len(hits)}")
    d = hits[0]
    meta = json.loads((d / f"{run_name}.json").read_text())
    trace = np.load(d / f"{run_name}_trace.npz")
    return {
        "run": run_name,
        "video": d / f"{run_name}-step-0.mp4",
        "env": meta["env_index"],
        "successes": trace["successes"][:, meta["env_index"]],
        "hz": float(meta["control_hz"]),
    }


def png_to_mp4(ffmpeg: str, png: Path, secs: float, out: Path) -> None:
    run([
        ffmpeg, "-y", "-v", "error",
        "-loop", "1", "-framerate", str(FPS), "-t", f"{secs:.2f}", "-i", str(png),
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out),
    ])


def make_title_card(png: Path, kicker: str, title: str, subtitle: str,
                    bullets: list[tuple[str, str, tuple]] | None = None, footer: str = "") -> None:
    img = Image.new("RGB", (W, H), NAVY)
    draw = ImageDraw.Draw(img)
    f_k = load_font(FONT_BOLD, 16)
    f_t = load_font(FONT_BOLD, 44)
    f_s = load_font(FONT_REG, 22)
    f_ch = load_font(FONT_BOLD, 20)
    f_cb = load_font(FONT_REG, 17)
    f_ft = load_font(FONT_REG, 16)

    draw.rectangle([0, 0, W, 8], fill=BLUE)
    draw.text((72, 92), kicker.upper(), fill=KICKER_BLUE, font=f_k)
    y = 132
    for line in title.split("\n"):
        draw.text((72, y), line, fill=WHITE, font=f_t)
        y += 56
    y += 14
    for line in subtitle.split("\n"):
        draw.text((72, y), line, fill=LIGHT_TEXT, font=f_s)
        y += 34

    if bullets:
        n = len(bullets)
        gap = 24
        cw = (W - 144 - gap * (n - 1)) // n
        cy0, cy1 = 385, 560
        for i, (head, body, accent) in enumerate(bullets):
            cx0 = 72 + i * (cw + gap)
            draw.rounded_rectangle([cx0, cy0, cx0 + cw, cy1], radius=12, fill=(32, 45, 76), outline=(58, 76, 118), width=2)
            draw.rectangle([cx0 + 18, cy0 + 20, cx0 + 26, cy0 + 48], fill=accent)
            draw.text((cx0 + 38, cy0 + 22), head, fill=WHITE, font=f_ch)
            by = cy0 + 66
            for bline in body.split("\n"):
                draw.text((cx0 + 20, by), bline, fill=LIGHT_TEXT, font=f_cb)
                by += 26

    if footer:
        draw.text((72, H - 56), footer, fill=KICKER_BLUE, font=f_ft)
    img.save(png)


def make_divider_card(ffmpeg: str, tag: str, title: str, sub: str, accent: tuple, secs: float, out: Path, tmp: Path) -> None:
    """Crisp full-screen transition divider card so switches between prompts/generalization tests are unmistakable."""
    png = tmp / f"{out.stem}.png"
    img = Image.new("RGB", (W, H), NAVY_DARK)
    draw = ImageDraw.Draw(img)
    f_tag = load_font(FONT_BOLD, 18)
    f_t = load_font(FONT_BOLD, 42)
    f_s = load_font(FONT_REG, 22)

    draw.rectangle([0, 0, W, 8], fill=accent)
    # Centered pill for tag
    tw = draw.textbbox((0, 0), tag, font=f_tag)[2] + 36
    draw.rounded_rectangle([(W - tw) // 2, 220, (W + tw) // 2, 260], radius=18, fill=accent)
    draw.text(((W - tw) // 2 + 18, 229), tag, fill=WHITE, font=f_tag)

    # Centered title & subtitle
    tb = draw.textbbox((0, 0), title, font=f_t)
    draw.text(((W - (tb[2] - tb[0])) // 2, 294), title, fill=WHITE, font=f_t)
    sb = draw.textbbox((0, 0), sub, font=f_s)
    draw.text(((W - (sb[2] - sb[0])) // 2, 366), sub, fill=LIGHT_TEXT, font=f_s)

    img.save(png)
    png_to_mp4(ffmpeg, png, secs, out)


def make_side_by_side_segment(ffmpeg: str, runs: list[str], labels: list[str], secs: float,
                              title: str, sub: str, out: Path, tmp: Path) -> None:
    cs = [load_clip(r) for r in runs]
    cmd = [ffmpeg, "-y", "-v", "error"]
    for c in cs:
        cmd += ["-t", f"{secs:.2f}", "-i", str(c["video"])]

    chains = []
    for idx, (c, lab) in enumerate(zip(cs, labels)):
        s = c["successes"]
        gained = np.cumsum(np.clip(np.diff(np.concatenate([[0.0], s])), 0, None))
        lines, last = [], None
        for i in range(min(len(gained), int(secs * c["hz"]))):
            if gained[i] != last:
                last = gained[i]
                lines.append(f"{(i + 1) / c['hz']:.4f} drawtext@cnt{idx} reinit text='goals reached {int(last)}';")
        cmds_path = tmp / f"{out.stem}_cmd{idx}.txt"
        cmds_path.write_text("\n".join(lines) + "\n")
        chains.append(
            f"[{idx}:v]fps={FPS},crop={CROP_W}:{CROP_H}:{CROP_X}:0,scale={TILE_W}:{TILE_H},"
            f"drawbox=x=0:y=ih-50:w=iw:h=50:color=0x17213A@0.78:t=fill,"
            f"drawtext=fontfile='{FONT_BOLD}':expansion=none:text='{esc(lab)}':x=18:y=h-38:fontsize=25:fontcolor=white,"
            f"sendcmd=f='{cmds_path}',"
            f"drawtext@cnt{idx}=fontfile='{FONT_REG}':expansion=none:text='goals reached 0':x=w-text_w-18:y=h-37:fontsize=23:fontcolor=0xD9E0EC"
            f"[t{idx}]"
        )

    gap = (W - 2 * TILE_W) // 3
    cap = (
        f"drawbox=x=0:y=0:w=iw:h=92:color=0x0F1628@0.95:t=fill,"
        f"drawtext=fontfile='{FONT_BOLD}':expansion=none:text='{esc(title)}':x=36:y=15:fontsize=31:fontcolor=white,"
        f"drawtext=fontfile='{FONT_REG}':expansion=none:text='{esc(sub)}':x=36:y=56:fontsize=22:fontcolor=0xD9E0EC"
    )
    vf = (
        ";".join(chains)
        + f";color=c=0x17213A:s={W}x{H}:r={FPS}:d={secs:.2f}[bg];"
        + f"[bg][t0]overlay=x={gap}:y={TOP}[m];[m][t1]overlay=x={2 * gap + TILE_W}:y={TOP},{cap}[v]"
    )
    cmd += ["-filter_complex", vf, "-map", "[v]", "-t", f"{secs:.2f}", "-r", str(FPS),
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)]
    run(cmd)


def make_task3_side_by_side(ffmpeg: str, names: list[str], labels: list[str], secs: float,
                            title: str, sub: str, out: Path, tmp: Path) -> None:
    """Side-by-side single-hand 6-colour Rubik's-cube Task 3 comparison matching Task 1 & 2 framing."""
    cmd = [ffmpeg, "-y", "-v", "error"]
    chains = []
    tile_s = 584
    for idx, (name, lab) in enumerate(zip(names, labels)):
        tr = json.loads((TASK3_DIR / f"{name}_trace.json").read_text())
        vid = TASK3_DIR / tr["video"]
        cmd += ["-t", f"{secs:.2f}", "-i", str(vid)]
        turns = tr["turns_per_step"]
        hz = float(tr["fps"])
        lines, last = [], None
        for i in range(min(len(turns), int(secs * hz))):
            if turns[i] != last:
                last = turns[i]
                lines.append(f"{(i + 1) / hz:.4f} drawtext@t3c{idx} reinit text='90° turns {int(last)}';")
        cmds_path = tmp / f"{out.stem}_t3cmd{idx}.txt"
        cmds_path.write_text("\n".join(lines) + "\n")
        chains.append(
            f"[{idx}:v]fps={FPS},scale={tile_s}:{tile_s},"
            f"drawbox=x=0:y=ih-50:w=iw:h=50:color=0x17213A@0.80:t=fill,"
            f"drawtext=fontfile='{FONT_BOLD}':expansion=none:text='{esc(lab)}':x=18:y=h-38:fontsize=24:fontcolor=white,"
            f"sendcmd=f='{cmds_path}',"
            f"drawtext@t3c{idx}=fontfile='{FONT_REG}':expansion=none:text='90° turns 0':x=w-text_w-18:y=h-37:fontsize=23:fontcolor=0xD9E0EC"
            f"[t{idx}]"
        )

    gap = (W - 2 * tile_s) // 3
    cap = (
        f"drawbox=x=0:y=0:w=iw:h=92:color=0x0F1628@0.95:t=fill,"
        f"drawtext=fontfile='{FONT_BOLD}':expansion=none:text='{esc(title)}':x=36:y=15:fontsize=31:fontcolor=white,"
        f"drawtext=fontfile='{FONT_REG}':expansion=none:text='{esc(sub)}':x=36:y=56:fontsize=21:fontcolor=0xD9E0EC"
    )
    vf = (
        ";".join(chains)
        + f";color=c=0x17213A:s={W}x{H}:r={FPS}:d={secs:.2f}[bg];"
        + f"[bg][t0]overlay=x={gap}:y={TOP}[m];[m][t1]overlay=x={2 * gap + tile_s}:y={TOP},{cap}[v]"
    )
    cmd += ["-filter_complex", vf, "-map", "[v]", "-t", f"{secs:.2f}", "-r", str(FPS),
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out)]
    run(cmd)


def make_staleness_widescreen(ffmpeg: str, out: Path) -> None:
    src = FINAL_PRES / "videos" / "slide08.mp4"
    title = "Act 2 · Why? Staleness—not rate—governs collapse"
    sub = "goal-4 at 2 Hz (left: real time then 1/4 slow-mo; right: error of last message vs 0.1 rad tolerance)"
    cap = (
        f"drawbox=x=0:y=0:w=iw:h=92:color=0x0F1628@0.95:t=fill,"
        f"drawtext=fontfile='{FONT_BOLD}':expansion=none:text='{esc(title)}':x=36:y=15:fontsize=31:fontcolor=white,"
        f"drawtext=fontfile='{FONT_REG}':expansion=none:text='{esc(sub)}':x=36:y=56:fontsize=21:fontcolor=0xD9E0EC"
    )
    vf = (
        f"[0:v]split=2[vtop][vbot];"
        f"[vtop]crop=616:584:0:0,scale=584:554[left];"
        f"[vbot]crop=616:464:0:584,scale=640:482[right];"
        f"color=c=0x17213A:s={W}x{H}:r={FPS}:d=16[bg];"
        f"[bg][left]overlay=x=24:y=124[m1];"
        f"[m1][right]overlay=x=624:y=160,{cap}[v]"
    )
    run([
        ffmpeg, "-y", "-v", "error", "-i", str(src),
        "-filter_complex", vf, "-map", "[v]", "-r", str(FPS),
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(out),
    ])


def concat_mp4s(ffmpeg: str, parts: list[Path], out: Path, tmp: Path) -> None:
    lst = tmp / f"{out.stem}_concat.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    run([
        ffmpeg, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19", "-r", str(FPS),
        "-movflags", "+faststart", str(out),
    ])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ffmpeg", required=True)
    ap.add_argument("--skip_vlm", action="store_true")
    args = ap.parse_args()
    ff = args.ffmpeg
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="gh_demo_"))

    # 1. Render VLM closed-loop clips (Standard + 3 Hardcore Zero-Shot Generalization tests)
    vlm_script = REPO / "Docs" / "Demo" / "make_vlm_closedloop_video.py"
    vlm_specs = [
        ("ep0", VLM_STD_DIR, "ep0_solve_L3_180deg", "vlm_end_to_end_ep0.mp4"),
        ("dark", VLM_SUITE_DIR, "gen3_dark_lowlight", "vlm_gen3_dark_lowlight.mp4"),
        ("revcol", VLM_SUITE_DIR, "gen4_ood_colour_rev", "vlm_gen4_ood_colour_rev.mp4"),
        ("spoken", VLM_GEN_DIR, "gen2_unseen_prompt", "vlm_gen2_unseen_prompt.mp4"),
    ]
    vlm_paths = {}
    for key, src_dir, prefix, fname in vlm_specs:
        out_ep = OUT_DIR / fname
        if not (args.skip_vlm and out_ep.exists()):
            print(f"Rendering VLM clip {prefix} -> {out_ep.name}...")
            run([
                sys.executable, str(vlm_script),
                "--ffmpeg", ff,
                "--input_dir", str(src_dir),
                "--prefix", prefix,
                "--cam", "demo",
                "--output", str(out_ep),
            ])
        vlm_paths[key] = out_ep

    # Helper divider cards for unmistakable transitions between VLM conditions
    div_std = tmp / "div_std.mp4"
    make_divider_card(ff, "DEMO 1 / 4  ·  STANDARD", "Nominal Illumination & Target Schema",
                      "100.0% real-time closed-loop (128/128 eps)  ·  3/3 optimal rounds", BLUE, 1.6, div_std, tmp)
    div_dark = tmp / "div_dark.mp4"
    make_divider_card(ff, "DEMO 2 / 4  ·  ZERO-SHOT DARK LOW-LIGHT", "10× Dimmer Illumination (Intensity 200)",
                      "100.0% zero-shot closed-loop (32/32 eps, 1.00× optimal) · 0 drops", AMBER, 1.6, div_dark, tmp)
    div_col = tmp / "div_col.mp4"
    make_divider_card(ff, "DEMO 3 / 4  ·  ZERO-SHOT OOD CUBE COLOURS", "Green & Blue Faces Physically Swapped",
                      "96.2% OOD rev-cycle / 93.7% novel-palette band acc.  ·  3/3 optimal", ORANGE, 1.6, div_col, tmp)
    div_prompt = tmp / "div_prompt.mp4"
    make_divider_card(ff, "DEMO 4 / 4  ·  CASUAL / SPOKEN PROMPT", "Unseen Conversational Instruction",
                      "Natural-language phrasing  →  100% visual state accuracy · 0 drops", GREEN, 1.6, div_prompt, tmp)

    # 2. Build GitHub README Teaser (Standard + Dark Low-Light + OOD Colours + Casual Prompt)
    print("Building GitHub Teaser (Standard + Dark Low-Light + OOD Colours + Casual Prompt)...")
    t_intro_png = tmp / "teaser_intro.png"
    make_title_card(
        t_intro_png,
        "End-to-End Visual Manipulation · Qwen3-VL-8B (SFT) → Unified v4 (60 Hz)",
        "Slow VLM Planner + Fast 60 Hz Dexterous Hand",
        "Pure 3-view RGB -> Map-then-Reason VLM -> Low-dim goal intent -> 60 Hz Unified v4 Shadow Hand\n"
        "Verified zero-shot generalization across 10× dark low-light, OOD cube colours, and spoken prompts",
        bullets=[
            ("1. Standard (128 eps)", "100.0% closed-loop success\n98.9% live 96-state acc.", BLUE),
            ("2. Dark Low-Light (32 eps)", "100.0% zero-shot (10× dimmer)\n1.00× optimal rounds · 0 drops", AMBER),
            ("3. OOD Colours & Prompts", "Green/blue swap 3/3 optimal\n93.7% novel-palette band acc.", GREEN),
        ],
        footer="NUS CS5478 · github.com/ziyouledi/CS5478_project",
    )
    t_intro = tmp / "teaser_intro.mp4"
    png_to_mp4(ff, t_intro_png, 3.0, t_intro)

    teaser_parts = [
        t_intro,
        div_std, vlm_paths["ep0"],
        div_dark, vlm_paths["dark"],
        div_col, vlm_paths["revcol"],
        div_prompt, vlm_paths["spoken"],
    ]
    teaser_out = OUT_DIR / "github_teaser.mp4"
    concat_mp4s(ff, teaser_parts, teaser_out, tmp)
    run([ff, "-y", "-v", "error", "-ss", "6.5", "-i", str(teaser_out), "-frames:v", "1", str(OUT_DIR / "github_teaser_poster.png")])
    print(f"Wrote {teaser_out}")

    # 3. Build Full GitHub End-to-End Demo Video (Act 0 - Act 5)
    print("Building Full GitHub End-to-End Demo Video (Act 0 - Act 5)...")
    parts = []

    # Act 0: Opening Title
    a0_png = tmp / "act0.png"
    make_title_card(
        a0_png,
        "Low-Dimensional Manipulation Intent Interfaces · NUS CS5478",
        "What Should a Slow Planner\nTell a Fast Dexterous Hand?",
        "Bridging 0.2–2 Hz reasoning models and 60 Hz 20-DOF Shadow Hand control across 3 Isaac Lab tasks",
        bullets=[
            ("Task 1 & 2", "SO(3) & Z-Axis Rotation\nContent, Rate, Delay & Staleness", BLUE),
            ("Task 3", "Rubik's-Style Hinged Cube\nUnified v4 + E3 10 Hz Reversal", ORANGE),
            ("VLM Closed-Loop", "Qwen3-VL-8B (SFT) -> Unified v4\nZero-Shot Lighting & Prompt Gen.", GREEN),
        ],
        footer="github.com/ziyouledi/CS5478_project",
    )
    a0 = tmp / "act0.mp4"
    png_to_mp4(ff, a0_png, 4.5, a0)
    parts.append(a0)

    # Act 1: Task 1 — Content, Rate Reversal, Latency
    a1a = tmp / "act1a.mp4"
    make_side_by_side_segment(
        ff, ["student-m1-blind-seed0", "student-m1-hand-8-seed0"],
        ["No intent (0% recovery)", "hand-8 · 60 Hz (87% recovery)"], 8.0,
        "Act 1 · Task 1 (Goal Hidden): At 60 Hz, 8-D action synergies recover 87%",
        "20-DOF Shadow Hand reorienting a cube in SO(3) without seeing the target orientation",
        a1a, tmp,
    )
    a1b = tmp / "act1b.mp4"
    make_side_by_side_segment(
        ff, ["student-m1-hand-8-hold30-seed0", "student-m1-goal-4-hold30-seed0"],
        ["2 Hz · hand-8 (3% recovery)", "2 Hz · goal-4 (44% recovery)"], 9.0,
        "Act 1 · Rate Reversal: At 2 Hz, action intents collapse while goals survive",
        "Trained under 2 Hz refresh (5 seeds): hand-8 drops 80% -> 3%; goal-4 degrades gently 78% -> 44%",
        a1b, tmp,
    )
    a1c = tmp / "act1c.mp4"
    make_side_by_side_segment(
        ff, ["student-m1-hand-8-seed0-delay6", "student-m1-goal-4-seed0-delay6"],
        ["hand-8 · 0.1 s delay (91% falls)", "goal-4 · 0.1 s delay (64% recovery)"], 8.0,
        "Act 1 · Latency Tolerance: 0.1 s delay breaks action intents, not goals",
        "Goal-level intents follow an analytical waiting-time law P(d) = 600 / (T0 + d)",
        a1c, tmp,
    )
    parts.extend([a1a, a1b, a1c])

    # Act 2: Staleness Mechanism + Task 2 Stale Goal Trap
    a2a = tmp / "act2a.mp4"
    make_staleness_widescreen(ff, a2a)
    a2b = tmp / "act2b.mp4"
    make_side_by_side_segment(
        ff, ["student-m2-hidden-goal-yaw-hold30-seed0", "student-m2-hidden-blind-seed0"],
        ["2 Hz · goal-yaw (16 goals/ep)", "No intent · blind spin (20 goals/ep)"], 8.0,
        "Act 2 · Task 2 (Z-Axis): A stale goal can be worse than no goal",
        "At 60 Hz goal-yaw reaches 36 goals/ep; at 2 Hz it falls behind a blind unidirectional spin heuristic",
        a2b, tmp,
    )
    parts.extend([a2a, a2b])

    # Act 3: Task 3 — Unified Single-Hand 6-Colour Rubik's Cube + E3 10 Hz Rate Reversal!
    a3a = tmp / "act3a.mp4"
    make_task3_side_by_side(
        ff, ["task3_e3_hand8_60hz", "task3_e3_hand8_10hz"],
        ["60 Hz · hand-8 (multi-skill)", "10 Hz · hand-8 (collapses to 4%)"], 8.0,
        "Act 3 · Task 3 (Rubik's-Style Hinged Cube): Multi-skill rate collapse",
        "Same single-hand high camera & 6-colour hinged cube: action intent hand-8 works at 60 Hz but stalls at 10 Hz",
        a3a, tmp,
    )
    a3b = tmp / "act3b.mp4"
    make_task3_side_by_side(
        ff, ["task3_e3_hand8_10hz", "task3_e3_goal7_10hz"],
        ["10 Hz · hand-8 (4% retained)", "10 Hz · goal-7 (69% retained)"], 8.0,
        "Act 3 · Task 3 Rate Reversal at 10 Hz: goal-7 beats hand-8 on multi-skill control",
        "Unified v4 teacher achieves 18.0 cycles (2.4× two-expert handoff) and 100% BFS Oracle across 96 states",
        a3b, tmp,
    )
    parts.extend([a3a, a3b])

    # Act 4: Real Upstream VLM Closed-Loop Control & Zero-Shot Generalization
    a4_intro_png = tmp / "act4_intro.png"
    make_title_card(
        a4_intro_png,
        "Act 4 · Bridging to Real Upstream VLMs (Plan C)",
        "End-to-End Visual Closed-Loop Control",
        "Qwen3-VL-8B-Instruct (Map-then-Reason SFT) driving frozen 60 Hz Unified v4\n"
        "100.0% real-time closed-loop success (128/128 eps) + zero-shot lighting & prompt generalization",
        bullets=[
            ("Canonical & OOD Colour", "98.7% canonical · 96.2% rev-cycle\n93.7% novel-palette band acc.", BLUE),
            ("Zero-Shot OOD Lighting", "Warm dim (650) & cool bright (3400)\nsolved in optimal rounds", AMBER),
            ("Real-Time Closed Loop", "Slow VLM (~3.5 s) + 60 Hz Hand\n0% drops · unseen NL prompts", GREEN),
        ],
    )
    a4_intro = tmp / "act4_intro.mp4"
    png_to_mp4(ff, a4_intro_png, 3.5, a4_intro)

    d_f1 = tmp / "df1.mp4"
    make_divider_card(ff, "CASE 1 / 3  ·  STANDARD", "End-to-End Closed-Loop Solving (L = 3)",
                      "100.0% real-time closed-loop (128/128 eps)  ·  3/3 optimal rounds", BLUE, 1.5, d_f1, tmp)
    d_f2 = tmp / "df2.mp4"
    make_divider_card(ff, "CASE 2 / 3  ·  ZERO-SHOT DARK LOW-LIGHT", "10× Dimmer Illumination (Intensity 200)",
                      "100.0% zero-shot closed-loop (32/32 eps, 1.00× optimal) · 0 drops", AMBER, 1.5, d_f2, tmp)
    d_f3 = tmp / "df3.mp4"
    make_divider_card(ff, "CASE 3 / 3  ·  ZERO-SHOT OOD CUBE COLOURS", "Green & Blue Faces Physically Swapped",
                      "96.2% OOD rev-cycle / 93.7% novel-palette band acc.  ·  3/3 optimal", ORANGE, 1.5, d_f3, tmp)

    parts.extend([
        a4_intro,
        d_f1, vlm_paths["ep0"],
        d_f2, vlm_paths["dark"],
        d_f3, vlm_paths["revcol"],
    ])

    # Act 5: Closing Summary Card
    a5_png = tmp / "act5.png"
    make_title_card(
        a5_png,
        "Key Takeaway",
        "An intent survives a slow channel only\nwhile its value stays valid between updates.",
        "Measure signal staleness against task tolerance before designing a planner-controller interface.",
        bullets=[
            ("Rate Reversal", "60 Hz: hand-8 (87%)\n2 Hz: goal-4 (44% vs 3%)", BLUE),
            ("Multi-Skill 60 Hz", "Unified v4: 2.4× handoff\n100% BFS · 0% pause drops", ORANGE),
            ("VLM Closed-Loop", "100% real-time (128/128)\nZero-shot lighting & prompts", GREEN),
        ],
        footer="Code, checkpoints & paper: https://github.com/ziyouledi/CS5478_project",
    )
    a5 = tmp / "act5.mp4"
    png_to_mp4(ff, a5_png, 5.0, a5)
    parts.append(a5)

    full_out = OUT_DIR / "github_full_demo.mp4"
    concat_mp4s(ff, parts, full_out, tmp)
    run([ff, "-y", "-v", "error", "-ss", "2.0", "-i", str(full_out), "-frames:v", "1", str(OUT_DIR / "github_full_demo_poster.png")])
    print(f"Wrote {full_out}")


if __name__ == "__main__":
    main()
