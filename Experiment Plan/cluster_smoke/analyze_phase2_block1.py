#!/usr/bin/env python3
"""Rate sweep phase 2, block 1: M2 task (goal = pure yaw), goal hidden, blind / goal-4 / goal-yaw x hold N.

Inputs: the 27 block 1 evaluation.json files (17, all 25 checkpoints, ZAxis task) and, optionally, the
phase 1 summary.json (Stage 1 goal-4 per N) for a side-by-side column.

* P_E = 31.13671875: M2 expert, final checkpoint, same evaluator protocol (`1032490`).
* P_B = mean final successes of the three m2-hidden blind runs (phase 2 definition 1).
* Recovery = (P - P_B) / (P_E - P_B), final checkpoint (iteration 5000).
* AUC = trapezoidal mean of successes / P_E over the 25 checkpoints; N80 = first iteration >= 0.8 P_E.
* Ceiling = min(1, (600 / N) / P_E) (one goal per refresh, derivation).
* Welch t (uncorrected, 3 seeds): vs the same intent at N = 1; goal-yaw vs goal-4 at the same N.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

from analyze_learning_curves import first_crossing, normalized_auc

P_E = 31.13671875
TASK = "Isaac-Repose-Cube-Shadow-ZAxis-Direct-v0"
EPISODE_STEPS = 600
CONDITIONS = [("blind", 1)] + [(i, n) for i in ("goal-4", "goal-yaw") for n in (1, 6, 12, 30)]
SEEDS = (0, 1, 2)


def load(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    record = report["intent_record"]
    points = sorted((c["iteration"], c["mean_successes_per_episode"], c["fall_fraction"]) for c in report["checkpoints"])
    it = np.array([p[0] for p in points], dtype=np.float64)
    val = np.array([p[1] for p in points], dtype=np.float64)
    return {
        "intent": record["intent"], "seed": int(record["seed"]), "source": path.name,
        "hold": int(report.get("intent_hold", 1) or 1), "hold_trained": int(record.get("intent_hold", 1) or 1),
        "task": report["task"], "train_task": record.get("task"), "goal_hidden": record.get("goal_hidden"),
        "delay": int(report.get("intent_delay", 0) or 0), "bits": int(report.get("intent_bits", 0) or 0),
        "iterations": it, "values": val, "successes": float(val[-1]), "fall": float(points[-1][2]),
    }


def welch(a, b):
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue) if len(a) > 1 and len(b) > 1 else None


def ms(values):
    a = np.asarray(values, dtype=np.float64)
    return float(a.mean()), (float(a.std(ddof=1)) if a.size > 1 else None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evals", type=Path, required=True)
    parser.add_argument("--phase1_summary", type=Path, help="results/rate-sweep-grid-39runs/summary.json")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    runs = [load(p) for p in sorted(args.evals.glob("*.json"))]
    failures = []
    for r in runs:
        if r["task"] != TASK or r["train_task"] != TASK:
            failures.append(f"{r['source']}: task {r['task']} / trained {r['train_task']}")
        if r["goal_hidden"] is not True:
            failures.append(f"{r['source']}: goal not hidden")
        if r["hold"] != r["hold_trained"] or r["delay"] or r["bits"]:
            failures.append(f"{r['source']}: eval channel differs from training")
        if r["iterations"].size != 25 or r["iterations"][-1] != 5000:
            failures.append(f"{r['source']}: {r['iterations'].size} checkpoints, last {r['iterations'][-1]}")
    groups: dict[tuple[str, int], list[dict]] = {}
    for r in runs:
        groups.setdefault((r["intent"], r["hold"]), []).append(r)
    for key in CONDITIONS:
        seeds = sorted(r["seed"] for r in groups.get(key, []))
        if seeds != list(SEEDS):
            failures.append(f"{key}: seeds {seeds}")
    for key in set(groups) - set(CONDITIONS):
        failures.append(f"unexpected condition {key}")

    p_b = float(np.mean([r["successes"] for r in groups.get(("blind", 1), [])] or [np.nan]))
    for r in runs:
        r["recovery"] = (r["successes"] - p_b) / (P_E - p_b)
        r["auc"] = normalized_auc(r["iterations"], r["values"], P_E)
        r["n80"] = first_crossing(r["iterations"], r["values"], 0.8 * P_E)
        r["ceiling"] = min(1.0, (EPISODE_STEPS / r["hold"]) / P_E)

    stage1 = {}
    if args.phase1_summary and args.phase1_summary.exists():
        for g in json.loads(args.phase1_summary.read_text(encoding="utf-8")).get("groups", []):
            if g.get("intent") == "goal-4":
                stage1[int(g["hold"])] = g.get("recovery_mean")

    summary = []
    for key in CONDITIONS:
        group = groups.get(key, [])
        if not group:
            continue
        intent, n = key
        rec, rec_sd = ms([r["recovery"] for r in group])
        auc, auc_sd = ms([r["auc"] for r in group])
        e = {
            "intent": intent, "hold": n, "rate_hz": 60.0 / n, "n_seeds": len(group),
            "successes_mean": ms([r["successes"] for r in group])[0],
            "recovery_mean": rec, "recovery_sd": rec_sd, "recovery_values": [r["recovery"] for r in sorted(group, key=lambda r: r["seed"])],
            "auc_mean": auc, "auc_sd": auc_sd, "ceiling": group[0]["ceiling"],
            "n80_reached": sum(r["n80"] is not None for r in group), "fall_mean": ms([r["fall"] for r in group])[0],
            "stage1_goal4_recovery": stage1.get(n) if intent != "blind" else None,
        }
        if intent != "blind" and n != 1:
            e["p_vs_same_intent_n1"] = welch([r["recovery"] for r in group], [r["recovery"] for r in groups.get((intent, 1), [])])
        if intent == "goal-yaw":
            e["p_vs_goal4_same_n"] = welch([r["recovery"] for r in group], [r["recovery"] for r in groups.get(("goal-4", n), [])])
        summary.append(e)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cols = ["intent", "hold", "seed", "successes", "recovery", "ceiling", "auc", "n80", "fall", "source"]
    with (args.output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        w = csv.writer(stream)
        w.writerow(cols)
        for r in sorted(runs, key=lambda r: (r["intent"], r["hold"], r["seed"])):
            w.writerow([r.get(c) for c in cols])

    def f(x, d=3):
        return "-" if x is None else f"{x:.{d}f}"

    print(f"P_E={P_E:.3f} P_B={p_b:.3f} (blind final successes: {[round(r['successes'], 3) for r in groups.get(('blind', 1), [])]})")
    for e in summary:
        print(f"{e['intent']:8} N={e['hold']:2d} ({e['rate_hz']:4.1f} Hz) P={e['successes_mean']:6.2f} "
              f"rec={f(e['recovery_mean'])}±{f(e['recovery_sd'])} ceil={e['ceiling']:.2f} auc={f(e['auc_mean'])} "
              f"n80={e['n80_reached']}/3 fall={e['fall_mean']:.3f} p_n1={f(e.get('p_vs_same_intent_n1'))} "
              f"p_vs_g4={f(e.get('p_vs_goal4_same_n'))} stage1_g4={f(e['stage1_goal4_recovery'])}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
        holds = [1, 6, 12, 30]
        rates = [60 / n for n in holds]
        for intent, color in (("goal-4", "#2a6fdb"), ("goal-yaw", "#e08a00")):
            es = [e for e in summary if e["intent"] == intent]
            axes[0].errorbar([e["rate_hz"] for e in es], [e["recovery_mean"] for e in es],
                             yerr=[e["recovery_sd"] or 0 for e in es], marker="o", capsize=3, color=color, label=f"M2 {intent}")
        if stage1:
            axes[0].plot([60 / n for n in holds if n in stage1], [stage1[n] for n in holds if n in stage1],
                         "o:", color="#2a6fdb", alpha=0.5, label="Stage 1 goal-4 (phase 1)")
        axes[0].plot(rates, [min(1.0, (EPISODE_STEPS / n) / P_E) for n in holds], "k--", lw=1, label="M2 ceiling")
        axes[0].set_xscale("log")
        axes[0].set_xticks(rates, [f"{r:g}" for r in rates])
        axes[0].set(xlabel="intent update rate (Hz)", ylabel="Recovery (final checkpoint)",
                    title="M2, goal hidden: Recovery vs update rate", ylim=(0, 1.05))
        axes[0].legend(fontsize=8)
        for (intent, n), color in zip([k for k in CONDITIONS], plt.cm.tab10.colors):
            group = groups.get((intent, n), [])
            if group:
                grid = group[0]["iterations"]
                stacked = np.vstack([np.interp(grid, r["iterations"], r["values"] / P_E) for r in group])
                axes[1].plot(grid, stacked.mean(0), color=color, ls="-" if intent != "goal-yaw" else "--",
                             label=f"{intent} N={n}")
        axes[1].set(xlabel="PPO iteration", ylabel="successes / P_E", title="Learning curves (mean of 3 seeds)")
        axes[1].legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(args.output_dir / "m2_recovery_vs_rate.png", dpi=150)
    except ImportError:
        print("matplotlib not available; no figure")

    status = "PASS" if not failures else "FAIL"
    (args.output_dir / "summary.json").write_text(json.dumps({
        "P_E": P_E, "P_E_source": "M2 expert ep_5000, job 1032490", "P_B": p_b, "P_B_source": "mean of m2-hidden blind seeds 0-2",
        "num_runs": len(runs), "status": status, "failures": failures, "groups": summary,
    }, indent=2) + "\n", encoding="utf-8")
    for x in failures:
        print("check_failure:", x)
    print(f"phase2_block1_analysis_status={status}")


if __name__ == "__main__":
    main()
