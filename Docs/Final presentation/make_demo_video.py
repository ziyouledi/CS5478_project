#!/usr/bin/env python3
"""Assemble the captioned demo video (1280x720, 30 fps, no audio) and its poster frame.

Usage: make_demo_video.py FFMPEG [--crop_test]
Clips: clips/<run>-job<JOB>_0/ for JOB in DEMO_JOBS (24_demo_videos.sbatch with VIDEO_ENV_INDEX=median,
16 envs, eye 0.30 -0.80 1.10, lookat -0.08 -0.40 0.55). Each clip films the env whose successes over the
10 s clip are closest to the median of the 16 envs (select_median_env.py), deterministic policy, seed
20261028. Every clip is cropped to the same window (neighbour hands out of frame) and shows a counter of
goals reached so far in the shown part, read from the clip's trace (successes of the filmed env).
Writes demo_video.mp4 and demo_poster.png next to this file. --crop_test writes crop_test.png instead
(first frame of every clip with the crop window drawn) to check the crop window.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CLIPS = HERE / "clips"
DEMO_JOBS = ("1125119", "1125120", "1125391",  # 1125391: student-m2-blind-seed0 (Task 2 expert)
             "1127897", "1127918",  # 1127897: hand-2 60 Hz, hand-8 2 Hz; 1127918: evaluation-only delay clips
             "1129142")  # 1129142: obj-4 2 Hz (slide 08)
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
CROP_W, CROP_H = 760, 720  # crop window in the 1280x720 source (not square: 40 px left margin for the goal cube)
CROP_X = 330          # default left edge of the crop (right edge stays at 1090); per run in CROP_X_RUN
TILE_H = 584          # displayed tile height; width keeps the crop aspect, even for libx264
TILE_W = 2 * round(CROP_W * TILE_H / CROP_H / 2)
CROP_X_RUN: dict[str, int] = {}
TOP = 104             # below the caption bar


def clip(run: str) -> dict:
    hits = [CLIPS / f"{run}-job{j}_0" for j in DEMO_JOBS if (CLIPS / f"{run}-job{j}_0").is_dir()]
    if len(hits) != 1:
        raise SystemExit(f"expected one clip dir for {run} in jobs {DEMO_JOBS}, found {len(hits)}")
    d = hits[0]
    meta = json.loads((d / f"{run}.json").read_text())
    trace = np.load(d / f"{run}_trace.npz")
    return {"run": run, "video": d / f"{run}-step-0.mp4", "env": meta["env_index"],
            "successes": trace["successes"][:, meta["env_index"]], "hz": float(meta["control_hz"])}


# (runs, seconds, title, subtitle, labels)
SEGMENTS = [
    (["student-stage1-blind-seed0", "student-m2-blind-seed0"], 8, "Expert (goal visible) on two tasks",
     "floating cube = target; Task 1: any orientation; Task 2: rotate about the vertical axis", ["Task 1", "Task 2"]),
    (["student-m1-blind-seed0", "student-m2-hidden-blind-seed0"], 8, "Goal hidden, no intent",
     "Task 1: holds the cube, cannot know where to turn it; Task 2: spins and passes the goals",
     ["Task 1", "Task 2"]),
    (["student-m1-hand-8-seed0", "student-m1-hand-8-hold6-seed0"], 9, "hand-8: 60 Hz vs 10 Hz",
     "8 action components; at 10 Hz (trained at 10 Hz) far fewer goals: Recovery 0.87 vs 0.18", ["60 Hz", "10 Hz"]),
    (["student-m1-goal-4-seed0", "student-m1-goal-4-hold30-seed0"], 9, "goal-4: 60 Hz vs 2 Hz",
     "the goal still works when sent twice a second: Recovery 0.77 vs 0.43", ["60 Hz", "2 Hz"]),
    (["student-m2-hidden-goal-yaw-seed0", "student-m2-hidden-goal-yaw-hold30-seed0"], 8, "Task 2, goal-yaw: 60 Hz vs 2 Hz",
     "at 2 Hz a stale goal is worse than none: 16 successes per episode vs 20 for spinning", ["60 Hz", "2 Hz"]),
]


def esc(s):
    return s.replace("\\", "\\\\").replace(":", "\\:").replace("'", "’").replace(",", "\\,")


def caption(title, sub):
    return (f"drawbox=x=0:y=0:w=iw:h=92:color=0x17213A@0.85:t=fill,"
            f"drawtext=fontfile='{FONT_BOLD}':text='{esc(title)}':x=36:y=16:fontsize=34:fontcolor=white,"
            f"drawtext=fontfile='{FONT}':text='{esc(sub)}':x=36:y=58:fontsize=24:fontcolor=0xD9E0EC")


def counter_cmds(c: dict, secs: float, name: str, path: Path) -> str:
    """sendcmd file: goals reached so far in the shown part (counter is zeroed on reset, so sum the increments).
    Video frame i (i/60 s) shows the state after step i, i.e. trace row i-1."""
    s = c["successes"]
    gained = np.cumsum(np.clip(np.diff(np.concatenate([[0.0], s])), 0, None))
    lines, last = [], None
    for i in range(min(len(gained), int(secs * c["hz"]))):
        if gained[i] != last:
            last = gained[i]
            lines.append(f"{(i + 1) / c['hz']:.4f} drawtext@{name} reinit text='goals reached {int(last)}';")
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def tile(idx, c, secs, label, tmp):
    """Filter chain for input idx: crop window, scale, bottom bar with label and live goal counter."""
    x = CROP_X_RUN.get(c["run"], CROP_X)
    cmds = counter_cmds(c, secs, f"cnt{idx}", tmp / f"cmd{idx}.txt")
    return (f"[{idx}:v]fps=30,crop={CROP_W}:{CROP_H}:{x}:0,scale={TILE_W}:{TILE_H},"
            f"drawbox=x=0:y=ih-50:w=iw:h=50:color=0x17213A@0.75:t=fill,"
            f"drawtext=fontfile='{FONT_BOLD}':text='{esc(label)}':x=18:y=h-38:fontsize=26:fontcolor=white,"
            f"sendcmd=f='{cmds}',"
            f"drawtext@cnt{idx}=fontfile='{FONT}':text='goals reached 0':x=w-text_w-18:y=h-37:fontsize=24:fontcolor=0xD9E0EC"
            f"[t{idx}]")


def run(cmd):
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if r.returncode:
        raise SystemExit(r.stderr[-3000:])


def crop_test(ffmpeg):
    tmp = Path(tempfile.mkdtemp())
    frames = []
    for seg in SEGMENTS:
        for run_name in seg[0]:
            c = clip(run_name)
            x = CROP_X_RUN.get(run_name, CROP_X)
            out = tmp / f"{run_name}.png"
            run([ffmpeg, "-y", "-ss", "2", "-i", str(c["video"]), "-frames:v", "1", "-vf",
                 f"drawbox=x={x}:y=0:w={CROP_W}:h={CROP_H}:color=red:t=6,scale=640:360,"
                 f"drawtext=fontfile='{FONT}':text='{esc(run_name)} env {c['env']}':x=10:y=10:fontsize=20:fontcolor=white:box=1:boxcolor=black",
                 str(out)])
            frames.append(out)
    inputs = sum((["-i", str(f)] for f in frames), [])
    n = len(frames)
    layout = "|".join(f"{(i % 2) * 640}_{(i // 2) * 360}" for i in range(n))
    run([ffmpeg, "-y", *inputs, "-filter_complex", f"xstack=inputs={n}:layout={layout}:fill=black",
         str(HERE / "crop_test.png")])
    print(f"wrote {HERE / 'crop_test.png'} ({n} clips)")


def main(ffmpeg):
    tmp = Path(tempfile.mkdtemp())
    parts = []
    for i, (runs, secs, title, sub, labels) in enumerate(SEGMENTS):
        cs = [clip(r) for r in runs]
        out = tmp / f"seg{i}.mp4"
        cmd = [ffmpeg, "-y"]
        for c in cs:
            cmd += ["-t", str(secs), "-i", str(c["video"])]
        chains = [tile(k, c, secs, labels[k], tmp) for k, c in enumerate(cs)]
        if len(cs) == 1:
            place = f"[bg][t0]overlay=x={(1280 - TILE_W) // 2}:y={TOP}"
        else:
            gap = (1280 - 2 * TILE_W) // 3
            place = f"[bg][t0]overlay=x={gap}:y={TOP}[m];[m][t1]overlay=x={2 * gap + TILE_W}:y={TOP}"
        vf = (";".join(chains) + f";color=c=0x17213A:s=1280x720:r=30:d={secs}[bg];"
              + place + f",{caption(title, sub)}[v]")
        cmd += ["-filter_complex", vf, "-map", "[v]", "-t", str(secs), "-an", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-crf", "20", str(out)]
        run(cmd)
        parts.append(out)
    lst = tmp / "list.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    video = HERE / "demo_video.mp4"
    run([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", "-movflags", "+faststart", str(video)])
    run([ffmpeg, "-y", "-ss", "3", "-i", str(video), "-frames:v", "1", str(HERE / "demo_poster.png")])
    total = sum(s[1] for s in SEGMENTS)
    print(f"wrote {video} ({total} s, {len(SEGMENTS)} segments) and demo_poster.png")


if __name__ == "__main__":
    if "--crop_test" in sys.argv[2:]:
        crop_test(sys.argv[1])
    else:
        main(sys.argv[1])
