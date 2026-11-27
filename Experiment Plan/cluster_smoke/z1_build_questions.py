#!/usr/bin/env python3
"""Build the Z1 question set from z1_capture.py frames (GOAL_LEVEL_RL_PLAN.md Z1). Standard library only.

Each question is one captured frame; the model answers two things a goal-level planner needs, both as colours:
  up         colour of the cube face pointing up: "white" (top layer up), "magenta" (bottom up), or the side
             colour red / green / yellow / blue of the bottom part on the upward side
  above_red  colour of the top layer's side that sits on top of the bottom part's red side; equals the detent
             (top layer turned k quarter turns about the cube's +z): k=0 red, 1 blue, 2 yellow, 3 green
Frames in motion are excluded (tilt of the up face > --max_tilt deg, or hinge > --max_residual deg from a detent),
because their ground truth is ambiguous. Sampling: up to --per_up frames per up colour, balanced over detents
in turn, with a fixed seed.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import random
from pathlib import Path

# z1_render_env.py: side colours by face direction in the link frame; bottom magenta, top white.
FACE_COLOUR = {"px": "red", "py": "green", "nx": "yellow", "ny": "blue", "pz": "white", "nz": "magenta"}
# Hinge +k quarter turns (counter-clockwise seen from +z) brings the top plate from direction rot(-k)(px) above px.
ABOVE_RED = ["red", "blue", "yellow", "green"]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--runs", required=True, help="Glob of z1_capture.py output dirs")
parser.add_argument("--output", required=True)
parser.add_argument("--per_up", type=int, default=50)
parser.add_argument("--max_tilt", type=float, default=25.0)
parser.add_argument("--max_residual", type=float, default=15.0)
parser.add_argument("--seed", type=int, default=20261119)
args = parser.parse_args()

rows = []
for run in sorted(glob.glob(args.runs)):
    for line in open(Path(run) / "frames.jsonl", encoding="utf-8"):
        row = json.loads(line)
        row["run_dir"] = run
        rows.append(row)
eligible = [r for r in rows if r["up_tilt_deg"] <= args.max_tilt and abs(r["detent_residual_deg"]) <= args.max_residual]
rng = random.Random(args.seed)
by_up = collections.defaultdict(list)
for r in eligible:
    by_up[FACE_COLOUR[r["up_face"]]].append(r)

questions = []
for up, group in sorted(by_up.items()):
    by_detent = collections.defaultdict(list)
    for r in group:
        by_detent[r["detent"]].append(r)
    for bucket in by_detent.values():
        rng.shuffle(bucket)
    picked = []
    while len(picked) < args.per_up and any(by_detent.values()):  # round-robin over detents
        for k in sorted(by_detent):
            if by_detent[k] and len(picked) < args.per_up:
                picked.append(by_detent[k].pop())
    for r in picked:
        cams = r["cameras"]
        questions.append({
            "id": f"{Path(r['run_dir']).name}/s{r['step']:04d}_e{r['env']:03d}",
            "run_dir": r["run_dir"],
            "step": r["step"],
            "env": r["env"],
            "answer": {"up": up, "above_red": ABOVE_RED[r["detent"]]},
            "kind": r["kind"],
            "up_tilt_deg": r["up_tilt_deg"],
            "detent": r["detent"],
            "visible_fraction": {c: v["cube_visible_fraction"] for c, v in cams.items()},
        })
rng.shuffle(questions)
Path(args.output).write_text("".join(json.dumps(q) + "\n" for q in questions), encoding="utf-8")
print(f"frames={len(rows)} eligible={len(eligible)} questions={len(questions)}")
print("up:", dict(collections.Counter(q["answer"]["up"] for q in questions)))
print("above_red:", dict(collections.Counter(q["answer"]["above_red"] for q in questions)))
