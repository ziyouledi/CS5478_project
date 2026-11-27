#!/usr/bin/env python3
"""Rate sweep phase 2, resume acceptance (b) (RATE_SWEEP_PLAN.md definition 3).

For each of the 9 block 2 resumed runs: ratio = mean P(5200, 5400, 5600) / P(ep_5000) of the
same run's phase 1 evaluation (P = mean successes per episode). Criterion: the mean ratio over
runs must not be more than 3% below 1 (mean ratio >= 0.97).

Usage: analyze_resume_transient.py PHASE1_DIR BLOCK2_DIR OUT_JSON
  PHASE1_DIR: phase 1 evaluation.json copies named <cond>-seed<s>-job<J>_<i>.json
  BLOCK2_DIR: block 2 evaluation.json copies named <cond>-seed<s>-resume-job<J>_<i>.json
"""
import json
import re
import statistics
import sys
from pathlib import Path

CHECKPOINTS = (5200, 5400, 5600)
THRESHOLD = 0.97
COMPARED_KEYS = ("task", "goal_hidden", "intent_hold", "deterministic", "seed",
                 "episodes_per_checkpoint", "episode_rule", "success_source")


def load(path):
    d = json.loads(Path(path).read_text())
    by_it = {c["iteration"]: c for c in d["checkpoints"]}
    return d, by_it


def main(phase1_dir, block2_dir, out_json):
    rows, errors = [], []
    for f in sorted(Path(block2_dir).glob("*-resume-job*.json")):
        m = re.match(r"(.+-seed\d)-resume-job", f.name)
        cond = m.group(1)
        src = sorted(Path(phase1_dir).glob(f"{cond}-job*.json"))
        if len(src) != 1:
            errors.append(f"{f.name}: {len(src)} phase 1 matches")
            continue
        d2, c2 = load(f)
        d1, c1 = load(src[0])
        for k in COMPARED_KEYS:
            if d1.get(k) != d2.get(k):
                errors.append(f"{f.name}: {k} differs ({d1.get(k)!r} vs {d2.get(k)!r})")
        missing = [it for it in CHECKPOINTS if it not in c2] + ([] if 5000 in c1 else ["phase1 5000"])
        if missing:
            errors.append(f"{f.name}: missing checkpoints {missing}")
            continue
        # Resume source must be the phase 1 run that was evaluated.
        run1 = Path(c1[5000]["checkpoint"]).parents[1].name
        if f"/{run1}/" not in json.dumps(d2.get("intent_record", {})) and run1 not in json.dumps(d2.get("intent_record", {})):
            errors.append(f"{f.name}: resume source not traced to {run1} in intent_record")
        p0 = c1[5000]["mean_successes_per_episode"]
        ps = [c2[it]["mean_successes_per_episode"] for it in CHECKPOINTS]
        last_it = max(c2)
        rows.append({
            "run": cond, "phase1_eval": src[0].name, "block2_eval": f.name,
            "P_5000_phase1": p0, "P_5200_5400_5600": ps,
            "ratio": statistics.mean(ps) / p0,
            "P_last": c2[last_it]["mean_successes_per_episode"], "last_iteration": last_it,
            "n_checkpoints": len(c2),
        })
    mean_ratio = statistics.mean(r["ratio"] for r in rows) if rows else float("nan")
    status = "PASS" if len(rows) == 9 and not errors and mean_ratio >= THRESHOLD else "FAIL"
    out = {"criterion": "mean over runs of mean P(5200,5400,5600)/P_phase1(5000) >= 0.97",
           "n_runs": len(rows), "mean_ratio": mean_ratio, "status": status,
           "errors": errors, "runs": rows}
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(out, indent=2))
    for r in rows:
        print(f"{r['run']:<22} P5000={r['P_5000_phase1']:6.2f}  P5200-5600="
              + " ".join(f"{p:6.2f}" for p in r["P_5200_5400_5600"])
              + f"  ratio={r['ratio']:.3f}  P{r['last_iteration']}={r['P_last']:6.2f} ({r['n_checkpoints']} ckpts)")
    for e in errors:
        print("ERROR", e)
    print(f"mean_ratio={mean_ratio:.4f}")
    print(f"resume_transient_status={status}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
