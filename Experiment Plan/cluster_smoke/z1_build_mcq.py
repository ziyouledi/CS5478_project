#!/usr/bin/env python3
"""Build the Z1 v2 frame set (standard multiple-choice evaluation, GOAL_LEVEL_RL_PLAN.md Z1 评测规范 step 2).
Standard library only.

Selects still frames from z1_capture.py runs and splits them into dev and test by episode (run, env), so frames
of one episode never appear in both. dev is for checking parsing and output format only; test is the reported
number. The question templates are fixed in z1_mcq_eval.py, not here.

Per frame it stores the ground truth for the three coarse-to-fine questions and the per-camera pixel counts that
z1_mcq_eval.py uses to decide whether a question is answerable from a given set of cameras:
  up     colour of the face pointing up (white / magenta / red / green / yellow / blue)
  turned whether the top layer is turned (detent != 0)
  band   colour of the top layer's strip on the red side: "none" if aligned, else blue / yellow / green
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
SIDES = ["px", "py", "nx", "ny"]  # counter-clockwise seen from +z
LABELS = [f"base_{s}" for s in SIDES] + ["base_nz"] + [f"top_{s}" for s in SIDES] + ["top_pz"]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--runs", required=True, help="Glob of z1_capture.py output dirs")
parser.add_argument("--output", required=True)
parser.add_argument("--dev_episodes", type=int, default=32, help="Episodes (run, env) held out as dev; rest is test")
parser.add_argument("--white_cap", type=float, default=0.35,
                    help="Cap white-up frames at this fraction of each split (white is 80%% of still frames)")
parser.add_argument("--max_tilt", type=float, default=25.0)
parser.add_argument("--max_residual", type=float, default=15.0)
parser.add_argument("--seed", type=int, default=20261120)
args = parser.parse_args()

rows = []
for run in sorted(glob.glob(args.runs)):
    for line in open(Path(run) / "frames.jsonl", encoding="utf-8"):
        row = json.loads(line)
        row["run_dir"] = run
        rows.append(row)
eligible = [r for r in rows if r["up_tilt_deg"] <= args.max_tilt and abs(r["detent_residual_deg"]) <= args.max_residual]
rng = random.Random(args.seed)
episodes = sorted({(r["run_dir"], r["env"]) for r in rows})
rng.shuffle(episodes)
dev = set(episodes[: args.dev_episodes])

frames = []
for split in ("dev", "test"):
    pool = [r for r in eligible if ((r["run_dir"], r["env"]) in dev) == (split == "dev")]
    white = [r for r in pool if r["up_face"] == "pz"]
    other = [r for r in pool if r["up_face"] != "pz"]
    n_white = min(len(white), round(args.white_cap / (1 - args.white_cap) * len(other)))
    by_detent = collections.defaultdict(list)
    for r in white:
        by_detent[r["detent"]].append(r)
    for bucket in by_detent.values():
        rng.shuffle(bucket)
    picked = []
    while len(picked) < n_white:  # round-robin over detents
        for k in sorted(by_detent):
            if by_detent[k] and len(picked) < n_white:
                picked.append(by_detent[k].pop())
    for r in sorted(other + picked, key=lambda r: (r["run_dir"], r["step"], r["env"])):
        k = r["detent"]
        frames.append({
            "id": f"{Path(r['run_dir']).name}/s{r['step']:04d}_e{r['env']:03d}",
            "split": split,
            "run_dir": r["run_dir"], "step": r["step"], "env": r["env"],
            "up_face": r["up_face"], "detent": k, "up_tilt_deg": r["up_tilt_deg"],
            "truth": {"up": FACE_COLOUR[r["up_face"]], "turned": "yes" if k else "no",
                      "band": "none" if k == 0 else FACE_COLOUR[SIDES[-k % 4]]},
            # top layer panel sitting on each bottom-part side (in the top link frame): top_{SIDES[i - k]}
            "above": {s: f"top_{SIDES[(i - k) % 4]}" for i, s in enumerate(SIDES)},
            "pixels": {cam: {mode: {lab: v[f"pixels_{mode}"].get(lab, 0) for lab in LABELS}
                             for mode in ("normal", "nohand")} for cam, v in r["cameras"].items()},
        })

Path(args.output).write_text("".join(json.dumps(f) + "\n" for f in frames), encoding="utf-8")
print(f"frames={len(rows)} eligible={len(eligible)} episodes={len(episodes)} dev_episodes={len(dev)}")
for split in ("dev", "test"):
    sub = [f for f in frames if f["split"] == split]
    print(f"{split}: n={len(sub)} up={dict(collections.Counter(f['truth']['up'] for f in sub))} "
          f"detent={dict(sorted(collections.Counter(f['detent'] for f in sub).items()))}")
