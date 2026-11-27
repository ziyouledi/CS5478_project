#!/usr/bin/env python3
"""Rate sweep phase 2 P1: Recovery under intent delay and quantisation (evaluation only).

Inputs: evaluation.json files of 23_eval_channel_sweep.sbatch (last checkpoint, 1024 envs), copied into one
directory. Expected: goal-4 N = 1 (seeds 0-2) at delay 0, 6, 30, 60, 120 and bits 8, 4, 3, 2;
hand-8 N = 1 (seeds 0-2) at delay 0, 6, 30, 60, 120. Delay 0 / bits 0 is the same-protocol baseline.
Recovery = (P - P_B) / (P_E - P_B) with the M1 constants. "retained" = P / P(baseline, same run).
Latency budget: the smallest tested delay at which mean Recovery is at most half the baseline mean.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

P_E = 26.1103515625
P_B = 0.006184895833333333
CONTROL_HZ = 60.0
EXPECTED = {
    "goal-4": [("delay", d) for d in (0, 6, 30, 60, 120)] + [("bits", b) for b in (8, 4, 3, 2)],
    "hand-8": [("delay", d) for d in (0, 6, 30, 60, 120)],
}
SEEDS = (0, 1, 2)


def load(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    record = report["intent_record"]
    last = report["checkpoints"][-1]
    delay = int(report.get("intent_delay", 0) or 0)
    bits = int(report.get("intent_bits", 0) or 0)
    return {
        "intent": record["intent"], "seed": int(record["seed"]), "run_dir": report["run_dir"],
        "trained_hold": int(record.get("intent_hold", 1)), "eval_hold": int(report.get("intent_hold", 1)),
        "last_only": bool(report.get("last_checkpoint_only")), "n_checkpoints": len(report["checkpoints"]),
        "delay": delay, "bits": bits, "kind": "bits" if bits else "delay", "value": bits if bits else delay,
        "iteration": last["iteration"], "successes": last["mean_successes_per_episode"],
        "fall": last["fall_fraction"], "source": path.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evals", type=Path, required=True, help="Directory of P1 evaluation.json files")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    rows = [load(p) for p in sorted(args.evals.glob("*.json"))]
    failures = []
    for r in rows:
        if r["delay"] and r["bits"]:
            failures.append(f"{r['source']}: delay and bits both set")
        if not r["last_only"] or r["n_checkpoints"] != 1 or r["iteration"] != 5000:
            failures.append(f"{r['source']}: not a single ep_5000 evaluation")
        if r["trained_hold"] != 1 or r["eval_hold"] != 1:
            failures.append(f"{r['source']}: hold != 1")
    index = {}
    for r in rows:
        key = (r["intent"], r["kind"], r["value"], r["seed"])
        if key in index:
            failures.append(f"duplicate {key}")
        index[key] = r
    expected_keys = {(i, k, v, s) for i, conds in EXPECTED.items() for k, v in conds for s in SEEDS}
    for key in sorted(expected_keys - set(index)):
        failures.append(f"missing {key}")
    for key in sorted(set(index) - expected_keys):
        failures.append(f"unexpected {key}")

    for r in rows:
        r["recovery"] = (r["successes"] - P_B) / (P_E - P_B)
        base = index.get((r["intent"], "delay", 0, r["seed"]))
        r["retained"] = r["successes"] / base["successes"] if base and base["successes"] > 0 else None
        r["delay_s"] = r["delay"] / CONTROL_HZ

    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["intent", "seed", "kind", "value", "delay_s", "successes", "recovery", "retained", "fall", "source"]
    rows.sort(key=lambda r: (r["intent"], r["kind"], r["value"], r["seed"]))
    with (args.output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for r in rows:
            writer.writerow([r.get(c) for c in columns])

    groups = {}
    for r in rows:
        groups.setdefault((r["intent"], r["kind"], r["value"]), []).append(r)
    summary = []
    for (intent, kind, value), group in sorted(groups.items()):
        rec = np.array([r["recovery"] for r in group])
        ret = [r["retained"] for r in group if r["retained"] is not None]
        summary.append({
            "intent": intent, "kind": kind, "value": value,
            "delay_s": value / CONTROL_HZ if kind == "delay" else 0.0, "n_seeds": len(group),
            "successes_mean": float(np.mean([r["successes"] for r in group])),
            "recovery_mean": float(rec.mean()), "recovery_std": float(rec.std(ddof=1)) if len(group) > 1 else None,
            "retained_mean": float(np.mean(ret)) if ret else None,
            "fall_mean": float(np.mean([r["fall"] for r in group])),
        })
        sd = summary[-1]["recovery_std"]
        print(f"{intent:7s} {kind:5s} {value:4d} recovery={rec.mean():.3f}{'' if sd is None else f' ± {sd:.3f}'}"
              f" retained={summary[-1]['retained_mean'] if ret else float('nan'):.3f}"
              f" fall={summary[-1]['fall_mean']:.3f} n={len(group)}")

    budgets = {}
    for intent in EXPECTED:
        curve = sorted((g["value"], g["recovery_mean"]) for g in summary if g["intent"] == intent and g["kind"] == "delay")
        base = dict(curve).get(0)
        half = next((d for d, rec in curve if base is not None and d > 0 and rec <= 0.5 * base), None)
        budgets[intent] = {"half_recovery_delay_steps": half,
                           "half_recovery_delay_s": None if half is None else half / CONTROL_HZ,
                           "note": "first tested delay with mean Recovery <= 0.5 x baseline; None = not reached"}
        print(f"{intent}: half-Recovery delay = {half} steps" + ("" if half is None else f" ({half / CONTROL_HZ:.2f} s)"))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        for intent, color in (("goal-4", "#2a6fdb"), ("hand-8", "#d9534f")):
            g = [s for s in summary if s["intent"] == intent and s["kind"] == "delay"]
            if g:
                axes[0].errorbar([s["delay_s"] for s in g], [s["recovery_mean"] for s in g],
                                 yerr=[s["recovery_std"] or 0 for s in g], marker="o", capsize=3, color=color, label=intent)
        axes[0].set_xlabel("intent delay (s)")
        axes[0].set_ylabel("Recovery (final checkpoint)")
        axes[0].set_title("Delay (N = 1, evaluation only)")
        axes[0].legend()
        g = [s for s in summary if s["intent"] == "goal-4" and (s["kind"] == "bits" or s["value"] == 0)]
        if g:
            labels = ["float" if s["kind"] == "delay" else f"{s['value']} bit" for s in g]
            order = sorted(range(len(g)), key=lambda i: -(g[i]["value"] if g[i]["kind"] == "bits" else 99))
            axes[1].bar([labels[i] for i in order], [g[i]["recovery_mean"] for i in order],
                        yerr=[g[i]["recovery_std"] or 0 for i in order], capsize=3, color="#2a6fdb")
        axes[1].set_ylabel("Recovery (final checkpoint)")
        axes[1].set_title("goal-4 quantisation (per component)")
        for ax in axes:
            ax.set_ylim(bottom=0)
        fig.tight_layout()
        fig.savefig(args.output_dir / "recovery_vs_delay_bits.png", dpi=150)
    except ImportError:
        print("matplotlib not available; no figure")

    status = "PASS" if not failures else "FAIL"
    (args.output_dir / "summary.json").write_text(json.dumps({
        "P_E": P_E, "P_B": P_B, "recovery_definition": "(P - P_B) / (P_E - P_B)",
        "retained_definition": "P / P(delay 0, bits off, same run, same protocol)",
        "status": status, "failures": failures, "latency_budget": budgets, "groups": summary,
    }, indent=2) + "\n", encoding="utf-8")
    for f in failures:
        print("check_failure:", f)
    print(f"channel_sweep_analysis_status={status}")


if __name__ == "__main__":
    main()
