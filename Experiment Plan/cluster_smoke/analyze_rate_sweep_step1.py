#!/usr/bin/env python3
"""Rate sweep step 1: summarise evaluation-only intent hold on M1 final checkpoints.

Inputs: evaluation.json files from 17_evaluate_students.sbatch with EVAL_INTENT_HOLD and EVAL_LAST_ONLY=1,
plus the M1 grid evaluations (all checkpoints, hold 1) for the reproduction check at N = 1.
Recovery = (P - P_B) / (P_E - P_B) with the M1 constants; ceiling = min(1, (600 / N) / P_E) (a goal
issued at most once per refresh; derivation from step 0, not a measurement).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

P_E = 26.1103515625
P_B = 0.006184895833333333
EPISODE_STEPS = 600
REPRO_TOLERANCE = 0.01  # relative; the evaluator order effect is ~0.5%


def load(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    record = report["intent_record"]
    last = report["checkpoints"][-1]
    return {
        "intent": record["intent"],
        "seed": int(record.get("seed", -1)),
        "run_dir": report["run_dir"],
        "hold": int(report.get("intent_hold", 1)),
        "iteration": last["iteration"],
        "successes": last["mean_successes_per_episode"],
        "fall": last["fall_fraction"],
    }


def seed_from_run_dir(run_dir: str) -> int:
    name = Path(run_dir).name
    return int(name.rsplit("seed", 1)[1].split("-")[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hold_evals", type=Path, required=True, help="Directory of step 1 evaluation.json files")
    parser.add_argument("--m1_evals", type=Path, required=True, help="Directory of M1 grid evaluation.json files")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.hold_evals.glob("*.json")):
        row = load(path)
        row["seed"] = seed_from_run_dir(row["run_dir"])
        row["source"] = path.name
        rows.append(row)
    m1 = {}
    for path in sorted(args.m1_evals.glob("*.json")):
        row = load(path)
        m1[row["run_dir"]] = row

    failures = []
    for row in rows:
        row["recovery"] = (row["successes"] - P_B) / (P_E - P_B)
        row["ceiling"] = min(1.0, (EPISODE_STEPS / row["hold"]) / P_E)
        ref = m1.get(row["run_dir"])
        row["m1_final_successes"] = ref["successes"] if ref else None
        if row["hold"] == 1:
            if ref is None or ref["iteration"] != row["iteration"]:
                failures.append(f"no matching M1 final checkpoint for {row['source']}")
            else:
                rel = abs(row["successes"] - ref["successes"]) / max(ref["successes"], 1e-9)
                row["m1_relative_diff"] = rel
                if rel > REPRO_TOLERANCE:
                    failures.append(f"{row['source']}: hold 1 {row['successes']:.3f} vs M1 {ref['successes']:.3f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["intent", "seed", "hold", "iteration", "successes", "recovery", "ceiling", "fall",
               "m1_final_successes", "source"]
    rows.sort(key=lambda r: (r["intent"], r["hold"], r["seed"]))
    with (args.output_dir / "hold_eval.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row.get(c) for c in columns])

    groups = {}
    for row in rows:
        groups.setdefault((row["intent"], row["hold"]), []).append(row)
    summary = []
    for (intent, hold), group in sorted(groups.items()):
        rec = np.array([r["recovery"] for r in group])
        summary.append({
            "intent": intent, "hold": hold, "rate_hz": 60.0 / hold, "n_seeds": len(group),
            "successes_mean": float(np.mean([r["successes"] for r in group])),
            "recovery_mean": float(rec.mean()),
            "recovery_std": float(rec.std(ddof=1)) if len(group) > 1 else None,
            "ceiling": group[0]["ceiling"],
            "fall_mean": float(np.mean([r["fall"] for r in group])),
        })
        print(f"{intent:8s} N={hold:4d} ({60 / hold:5.1f} Hz) recovery={rec.mean():.3f}"
              f"{'' if len(group) < 2 else f' ± {rec.std(ddof=1):.3f}'} ceiling={group[0]['ceiling']:.2f} n={len(group)}")
    status = "PASS" if not failures else "FAIL"
    (args.output_dir / "summary.json").write_text(json.dumps({
        "P_E": P_E, "P_B": P_B, "recovery_definition": "(P - P_B) / (P_E - P_B)",
        "ceiling_definition": "min(1, (600 / N) / P_E)", "repro_tolerance": REPRO_TOLERANCE,
        "repro_failures": failures, "status": status, "groups": summary,
    }, indent=2) + "\n", encoding="utf-8")
    for failure in failures:
        print("repro_failure:", failure)
    print(f"rate_sweep_step1_status={status}")


if __name__ == "__main__":
    main()
