#!/usr/bin/env python3
"""Per-slide result videos for the v2 deck: videos/slideNN.mp4 + videos/slideNN_poster.png (NN = 05-10).

Usage: make_slide_videos.py FFMPEG [NN ...]
Each video = pair A (8 s) then pair B (8 s), no audio, 30 fps. Slides 05/06/07/10: two tiles side by side
(1232x584); slides 08/09: two tiles stacked (09: 616x1168; 08: video 616x584 over a 616x464 timeline), matching build_final_slides.py. Tiles reuse the crop,
label bar and live "goals reached" counter of make_demo_video.py (clips and DEMO_JOBS there).
Slide 08: top = goal-4 at 2 Hz (A: real time, B: a 2 s window at 1/4 speed), bottom = timeline from the clip's
trace: angle between the message last sent (refresh every 30 steps) and the current value, for the goal and (B) for
an obj-4 waypoint 0.5 rad ahead on the same cube motion, against the 0.1 rad success tolerance; cursor in sync.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from make_demo_video import FONT, FONT_BOLD, CROP_W, CROP_H, CROP_X, CROP_X_RUN, TILE_W, TILE_H, clip, esc, run

HERE = Path(__file__).resolve().parent
OUT = HERE / "videos"
SEG = 8          # seconds per pair
FPS = 30
HOLD = 30        # goal-4 2 Hz: intent refreshed when the episode step is a multiple of 30
SLOW = 4         # slide 08 B: 1/4 speed -> a 2 s window
TL_LEFT, TL_RIGHT = 0.13, 0.95  # timeline axes, fraction of tile width
TL_TILE_H = 464                 # slide 08 timeline tile height (even): video 616x1048 fits the user-resized box (aspect 0.587)
TL_BOT, TL_H = 0.19, 0.53       # timeline axes, fraction of timeline tile height

# NN: (layout, pair A [(run, label)] x2, pair B)
SLIDES = {
    5: ("side", [("student-stage1-blind-seed0", "Expert · Task 1"), ("student-m2-blind-seed0", "Expert · Task 2")],
              [("student-m1-blind-seed0", "No intent · Task 1"), ("student-m2-hidden-blind-seed0", "No intent · Task 2")]),
    6: ("side", [("student-m1-blind-seed0", "No intent"), ("student-m1-hand-8-seed0", "hand-8")],
              [("student-m1-hand-2-seed0", "hand-2"), ("student-m1-hand-8-seed0", "hand-8")]),
    7: ("side", [("student-m1-hand-8-seed0", "60 Hz · hand-8"), ("student-m1-goal-4-seed0", "60 Hz · goal-4")],
              [("student-m1-hand-8-hold30-seed0", "2 Hz · hand-8"), ("student-m1-goal-4-hold30-seed0", "2 Hz · goal-4")]),
    9: ("stack", [("student-m1-hand-8-seed0-delay0", "hand-8 · no delay"),
                  ("student-m1-hand-8-seed0-delay6", "hand-8 · 0.1 s delay")],
               [("student-m1-goal-4-seed0-delay6", "goal-4 · 0.1 s delay"),
                ("student-m1-goal-4-seed0-delay60", "goal-4 · 1 s delay")]),
    10: ("side", [("student-m2-hidden-goal-yaw-seed0", "60 Hz · goal-yaw"), ("student-m2-hidden-blind-seed0", "No intent")],
               [("student-m2-hidden-goal-yaw-hold30-seed0", "2 Hz · goal-yaw"), ("student-m2-hidden-blind-seed0", "No intent")]),
}
# Slide 08: one goal-4 clip at 2 Hz (A real time, B a 2 s window at 1/4 speed). The timeline shows how wrong the
# message last sent is, for the goal (what this student got) and, on the same cube motion, for an obj-4 waypoint
# 0.5 rad ahead (computed from the trace; the obj-4 student itself is not shown).
SLIDE08_RUN = "student-m1-goal-4-hold30-seed0"
OBJ_LOOKAHEAD = 0.5  # rad, obj-4 lookahead of the M1 grid
TOL = 0.1        # rad, the environment's success tolerance


def tile(idx, c, label, start, shown, speed, tmp):
    """Input idx (already trimmed to start..start+shown) -> cropped tile with label and counter of goals reached
    since `start`. Output plays shown*speed seconds."""
    hz, s = c["hz"], c["successes"]
    gained = np.cumsum(np.clip(np.diff(np.concatenate([[0.0], s])), 0, None))
    i0 = int(round(start * hz))
    base = gained[i0 - 1] if i0 > 0 else 0.0
    lines, last = [], None
    for i in range(i0, min(len(gained), i0 + int(shown * hz))):
        g = gained[i] - base
        if g != last:
            last = g
            lines.append(f"{((i + 1) / hz - start) * speed:.4f} drawtext@cnt{idx} reinit text='goals reached {int(g)}';")
    cmds = tmp / f"cmd{idx}.txt"
    cmds.write_text("\n".join(lines) + "\n")
    x = CROP_X_RUN.get(c["run"], CROP_X)
    return (f"[{idx}:v]setpts={speed}*(PTS-STARTPTS),fps={FPS},crop={CROP_W}:{CROP_H}:{x}:0,scale={TILE_W}:{TILE_H},"
            f"drawbox=x=0:y=ih-50:w=iw:h=50:color=0x17213A@0.75:t=fill,"
            f"drawtext=fontfile='{FONT_BOLD}':text='{esc(label)}':x=18:y=h-38:fontsize=26:fontcolor=white,"
            f"sendcmd=f='{cmds}',"
            f"drawtext@cnt{idx}=fontfile='{FONT}':text='goals reached 0':x=w-text_w-18:y=h-37:fontsize=24:fontcolor=0xD9E0EC"
            f"[t{idx}]")


def segment(ffmpeg, tiles, layout, out, tmp):
    """tiles: list of (clip, label, start, shown, speed) or ('png', path, t0, span) timeline tiles."""
    cmd, chains = [ffmpeg, "-y"], []
    for k, t in enumerate(tiles):
        if t[0] == "png":
            _, png, cursor = t
            cmd += ["-loop", "1", "-framerate", str(FPS), "-t", str(SEG), "-i", str(png)]
            x0, w = TL_LEFT * TILE_W, (TL_RIGHT - TL_LEFT) * TILE_W
            chains.append(f"color=c=0xD62728:s=3x{cursor[1] - cursor[0]}:r={FPS}:d={SEG}[cur{k}];"
                          f"[{k}:v]format=yuv420p[p{k}];[p{k}][cur{k}]overlay=x='{x0:.1f}+{w:.1f}*t/{SEG}':y={cursor[0]}[t{k}]")
        else:
            c, label, start, shown, speed = t
            cmd += ["-ss", f"{start:.4f}", "-t", f"{shown:.4f}", "-i", str(c["video"])]
            chains.append(tile(k, c, label, start, shown, speed, tmp))
    stackf = "hstack" if layout == "side" else "vstack"
    vf = ";".join(chains) + f";[t0][t1]{stackf}=inputs=2,format=yuv420p[v]"
    cmd += ["-filter_complex", vf, "-map", "[v]", "-t", str(SEG), "-r", str(FPS), "-an", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-crf", "20", str(out)]
    run(cmd)


def concat(ffmpeg, parts, nn):
    lst = parts[0].parent / f"list{nn}.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    video = OUT / f"slide{nn:02d}.mp4"
    run([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", "-movflags", "+faststart", str(video)])
    run([ffmpeg, "-y", "-ss", "3", "-i", str(video), "-frames:v", "1", str(OUT / f"slide{nn:02d}_poster.png")])
    print(f"wrote {video}")


def rot_dist(a, b):
    return 2 * np.arccos(np.clip(np.abs((a * b).sum(-1)), 0, 1))


def slerp(q0, q1, f):
    """numpy copy of intent_modules.quat_slerp (shortest path)."""
    q0 = q0 / np.linalg.norm(q0, axis=-1, keepdims=True)
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    dot = (q0 * q1).sum(-1)
    q1 = np.where((dot < 0)[:, None], -q1, q1)
    om = np.arccos(np.clip(np.abs(dot), 0, 1))
    so = np.sin(om)
    small = so < 1e-6
    safe = np.where(small, 1.0, so)
    w0 = np.where(small, 1 - f, np.sin((1 - f) * om) / safe)
    w1 = np.where(small, f, np.sin(f * om) / safe)
    out = w0[:, None] * q0 + w1[:, None] * q1
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def held(target, ep):
    """Intent as the student sees it: refreshed when the episode step is a multiple of HOLD.
    Row i = state after step i; the observation for step i+1 has episode step ep[i]."""
    told = np.empty_like(target)
    refresh = np.zeros(len(ep), bool)
    cur = target[0]
    for i in range(len(ep)):
        if ep[i] % HOLD == 0:
            cur, refresh[i] = target[i], True
        told[i] = cur
    return told, refresh


def slide08_trace():
    c = clip(SLIDE08_RUN)
    d = c["video"].parent
    tr = np.load(d / f"{SLIDE08_RUN}_trace.npz")
    rec = json.loads((d / f"{SLIDE08_RUN}.json").read_text())["intent_record"]
    assert rec["intent"] == "goal-4" and rec["intent_hold"] == HOLD, rec
    errs, fracs = {}, {}
    for e in range(tr["object_rot"].shape[1]):
        obj, goal, ep = tr["object_rot"][:, e], tr["goal_rot"][:, e], tr["episode_step"][:, e]
        theta = rot_dist(obj, goal)
        way = slerp(obj, goal, np.minimum(1.0, OBJ_LOOKAHEAD / np.maximum(theta, 1e-6)))
        for name, target in (("goal", goal), ("obj", way)):
            told, refresh = held(target, ep)
            err = rot_dist(told, target)
            fracs.setdefault(name, []).append(float((err > TOL).mean()))
            if e == c["env"]:
                errs[name], ref = err, refresh
    reached = np.where(np.diff(np.concatenate([[0.0], c["successes"]])) > 0)[0]
    t = (np.arange(len(ref)) + 1) / c["hz"]
    return c, t, errs, ref, reached, {k: float(np.mean(v)) for k, v in fracs.items()}


SERIES = (("goal", "goal (what this hand gets)", "#2B5BD7"), ("obj", "waypoint 0.5 rad ahead (obj-4)", "#D9480F"))


def timeline_png(path, t, errs, names, refresh, reached, t0, span, title, sub, ymax=1.0):
    W, H = (TILE_W + 0.01) / 100, (TL_TILE_H + 0.01) / 100  # +0.01: avoid float floor to 463 px
    fig = plt.figure(figsize=(W, H), dpi=100, facecolor="white")
    ax = fig.add_axes([TL_LEFT, TL_BOT, TL_RIGHT - TL_LEFT, TL_H])
    m = (t >= t0) & (t <= t0 + span)
    for i in np.where(refresh & m)[0]:
        ax.axvline(t[i], color="#888888", lw=1, ls=":", alpha=0.9)
    ax.axhline(TOL, color="#555555", lw=1.2, ls="--")
    ax.text(t0 + 0.99 * span, TOL + 0.02, "success tolerance 0.1 rad", fontsize=9.5, color="#555555",
            va="bottom", ha="right")
    for key, label, color in SERIES:
        if key in names:
            ax.plot(t[m], np.minimum(errs[key][m], ymax), color=color, lw=2.2, drawstyle="steps-post", label=label)
    for r in reached:
        if t0 <= t[r] <= t0 + span:
            ax.plot(t[r], 0, marker="v", color="#2CA02C", ms=11, clip_on=False, zorder=5)
    ax.set_xlim(t0, t0 + span)
    ax.set_ylim(0, ymax)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "≥1"])
    ax.set_xlabel("time (s)", fontsize=13)
    ax.set_ylabel("error (rad)", fontsize=13)
    ax.tick_params(labelsize=11)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.text(0.04, 0.965, title, fontsize=15, weight="bold", color="#17213A", va="top")
    fig.text(0.04, 0.895, sub, fontsize=11, color="#555555", va="top")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=1, fontsize=10.5, frameon=False, handlelength=1.8)
    fig.text(0.04, 0.025, "dotted: message re-sent (2 Hz) · ▼ goal reached", fontsize=10.5, color="#555555")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    with Image.open(path) as im:
        assert im.size == (TILE_W, TL_TILE_H), im.size
    # cursor rows in pixels (axes top..bottom)
    return int(round((1 - TL_BOT - TL_H) * TL_TILE_H)), int(round((1 - TL_BOT) * TL_TILE_H))


def slide08(ffmpeg, tmp):
    c, t, errs, refresh, reached, fracs = slide08_trace()
    span = SEG / SLOW
    win = int(span * c["hz"])
    # B window (half-second grid, start 1..8 s): the 2 s whose wrong-message shares are closest to the filmed
    # episode's, for goal and waypoint together (representative, not cherry-picked); ties -> more goals reached
    wrong = {k: errs[k] > TOL for k in errs}
    def score(b):
        i = int(round(b * c["hz"]))
        dev = sum(abs(wrong[k][i:i + win].mean() - wrong[k].mean()) for k in wrong)
        return (round(dev, 3), -int(((reached >= i) & (reached < i + win)).sum()))
    b0 = min(np.arange(1.0, 10 - span + 0.01, 0.5), key=score)
    pct = {k: f"{v:.0%}" for k, v in fracs.items()}
    parts = []
    for name, (t0, sp, speed, lab, names, title, sub) in {
        "A": (0.0, SEG, 1, "goal-4 · 2 Hz", ("goal",), "How wrong is the last message?",
              "goal sent every 0.5 s · wrong only after a goal is reached"),
        "B": (b0, span, SLOW, "goal-4 · 2 Hz · ¼ speed", ("goal", "obj"), "Same motion, same 2 Hz",
              f"message wrong (> 0.1 rad): goal {pct['goal']} vs waypoint {pct['obj']} of the time"),
    }.items():
        png = tmp / f"tl08{name}.png"
        cursor = timeline_png(png, t, errs, names, refresh, reached, t0, sp, title, sub)
        out = tmp / f"s08{name}.mp4"
        segment(ffmpeg, [(c, lab, t0, sp, speed), ("png", png, cursor)], "stack", out, tmp)
        parts.append(out)
    concat(ffmpeg, parts, 8)
    i0 = int(b0 * c["hz"])
    print(f"  slide 08 B window {b0:.1f}-{b0 + span:.1f} s; 16-env share of steps wrong: {pct}; "
          f"filmed env: goal {(errs['goal'] > TOL).mean():.0%}, obj {(errs['obj'] > TOL).mean():.0%}; "
          f"in window: goal {(errs['goal'][i0:i0 + win] > TOL).mean():.0%}, obj {(errs['obj'][i0:i0 + win] > TOL).mean():.0%}")


def main(ffmpeg, which):
    OUT.mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp())
    for nn in which:
        if nn == 8:
            slide08(ffmpeg, tmp)
            continue
        layout, a, b = SLIDES[nn]
        parts = []
        for name, pair in (("A", a), ("B", b)):
            out = tmp / f"s{nn:02d}{name}.mp4"
            segment(ffmpeg, [(clip(r), lab, 0.0, SEG, 1) for r, lab in pair], layout, out, tmp)
            parts.append(out)
        concat(ffmpeg, parts, nn)


if __name__ == "__main__":
    nums = [int(x) for x in sys.argv[2:]] or [5, 6, 7, 8, 9, 10]
    main(sys.argv[1], nums)
