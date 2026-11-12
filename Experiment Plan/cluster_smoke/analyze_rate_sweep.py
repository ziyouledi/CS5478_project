#!/usr/bin/env python3
"""Rate sweep step 3: Recovery, AUC and N80 per (intent, hold N) for students trained under hold.

Inputs: the 39 grid evaluation.json files (goal-4 at N = 1; hand-2, hand-8, obj-4, goal-4 at N = 6, 12, 30;
seeds 0-2) and the M1 grid evaluations, which give hand-2, hand-8 and obj-4 at N = 1.
Optionally the step 1 hold_eval.csv (evaluation-only hold on the 60 Hz students) for comparison.

* Recovery = (P - P_B) / (P_E - P_B) on the final checkpoint, M1 constants.
* AUC = trapezoidal mean of successes / P_E over the evaluated checkpoints (as analyze_learning_curves.py).
* N80 = first iteration with successes >= 0.8 P_E, interpolated; None if never reached.
* Ceiling = min(1, (600 / N) / P_E): at most one goal per refresh (step 0 derivation, not a measurement).
* Tests: Welch t (uncorrected, 3 seeds) against the same intent at N = 1, and against goal-4 at the same N.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

from analyze_learning_curves import first_crossing, normalized_auc

P_E = 26.1103515625
P_B = 0.006184895833333333
EPISODE_STEPS = 600
INTENTS = ["hand-2", "hand-8", "obj-4", "goal-4"]
HOLDS = [1, 6, 12, 30]
EXPECTED_CHECKPOINTS = 25
EXPECTED_FINAL_ITERATION = 5000


def load(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    record = report["intent_record"]
    points = sorted((c["iteration"], c["mean_successes_per_episode"], c["fall_fraction"]) for c in report["checkpoints"])
    iterations = np.array([p[0] for p in points], dtype=np.float64)
    values = np.array([p[1] for p in points], dtype=np.float64)
    hold = int(report.get("intent_hold", record.get("intent_hold", 1)) or 1)
    trained = int(report.get("intent_hold_trained", record.get("intent_hold", 1)) or 1)
    return {
        "intent": record["intent"], "seed": int(record["seed"]), "hold": hold, "hold_trained": trained,
        "source": path.name, "run_dir": report["run_dir"],
        "iterations": iterations, "values": values,
        "checkpoints": int(iterations.size), "final_iteration": float(iterations[-1]),
        "successes": float(values[-1]), "fall": float(points[-1][2]),
    }


def welch(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2 or len(b) < 2:
        return None
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def mean_sd(values: list[float]) -> tuple[float, float | None]:
    arr = np.array(values, dtype=np.float64)
    return float(arr.mean()), (float(arr.std(ddof=1)) if arr.size > 1 else None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid_evals", type=Path, required=True, help="Directory of the 39 grid evaluation.json files")
    parser.add_argument("--m1_evals", type=Path, required=True, help="Directory of M1 grid evaluation.json files")
    parser.add_argument("--step1_csv", type=Path, help="hold_eval.csv of step 1 (evaluation-only hold)")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--skip_plot", action="store_true")
    args = parser.parse_args()

    runs = [load(p) for p in sorted(args.grid_evals.glob("*.json"))]
    runs += [r for r in (load(p) for p in sorted(args.m1_evals.glob("*.json")))
             if r["intent"] in INTENTS and "gate" not in r["source"]]

    failures = []
    for r in runs:
        if r["hold"] != r["hold_trained"]:
            failures.append(f"{r['source']}: evaluated hold {r['hold']} != trained hold {r['hold_trained']}")
        if r["checkpoints"] != EXPECTED_CHECKPOINTS or r["final_iteration"] != EXPECTED_FINAL_ITERATION:
            failures.append(f"{r['source']}: {r['checkpoints']} checkpoints, final iteration {r['final_iteration']}")
        r["recovery"] = (r["successes"] - P_B) / (P_E - P_B)
        r["auc"] = normalized_auc(r["iterations"], r["values"], P_E)
        r["n80"] = first_crossing(r["iterations"], r["values"], 0.8 * P_E)
        r["ceiling"] = min(1.0, (EPISODE_STEPS / r["hold"]) / P_E)

    groups: dict[tuple[str, int], list[dict]] = {}
    for r in runs:
        groups.setdefault((r["intent"], r["hold"]), []).append(r)
    for intent in INTENTS:
        for hold in HOLDS:
            seeds = sorted(r["seed"] for r in groups.get((intent, hold), []))
            if seeds != [0, 1, 2]:
                failures.append(f"{intent} N={hold}: seeds {seeds}, expected [0, 1, 2]")
    extra = sorted(set(groups) - {(i, h) for i in INTENTS for h in HOLDS})
    if extra:
        failures.append(f"unexpected groups {extra}")

    step1 = {}
    if args.step1_csv and args.step1_csv.exists():
        for row in csv.DictReader(args.step1_csv.open(encoding="utf-8")):
            step1.setdefault((row["intent"], int(row["hold"])), []).append(float(row["recovery"]))

    summary = []
    for intent in INTENTS:
        for hold in HOLDS:
            group = sorted(groups.get((intent, hold), []), key=lambda r: r["seed"])
            if not group:
                continue
            base = groups.get((intent, 1), [])
            goal = groups.get(("goal-4", hold), [])
            rec, rec_sd = mean_sd([r["recovery"] for r in group])
            auc, auc_sd = mean_sd([r["auc"] for r in group])
            reached = [r["n80"] for r in group if r["n80"] is not None]
            entry = {
                "intent": intent, "hold": hold, "rate_hz": 60.0 / hold, "seeds": [r["seed"] for r in group],
                "successes": [r["successes"] for r in group],
                "recovery_values": [r["recovery"] for r in group],
                "recovery_mean": rec, "recovery_sd": rec_sd,
                "ceiling": group[0]["ceiling"], "recovery_over_ceiling": rec / group[0]["ceiling"],
                "auc_mean": auc, "auc_sd": auc_sd,
                "n80_values": [r["n80"] for r in group], "n_reached_80": len(reached),
                "n80_mean_reached": float(np.mean(reached)) if reached else None,
                "fall_mean": float(np.mean([r["fall"] for r in group])),
                "step1_eval_only_recovery_mean": float(np.mean(step1[(intent, hold)])) if (intent, hold) in step1 else None,
            }
            if hold != 1:
                entry["p_recovery_vs_same_intent_n1"] = welch(entry["recovery_values"], [r["recovery"] for r in base])
                entry["p_auc_vs_same_intent_n1"] = welch([r["auc"] for r in group], [r["auc"] for r in base])
            if intent != "goal-4":
                entry["p_recovery_vs_goal4_same_n"] = welch(entry["recovery_values"], [r["recovery"] for r in goal])
                entry["p_auc_vs_goal4_same_n"] = welch([r["auc"] for r in group], [r["auc"] for r in goal])
            summary.append(entry)

    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    columns = ["intent", "hold", "seed", "successes", "recovery", "ceiling", "auc", "n80", "fall", "source"]
    with (out / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for r in sorted(runs, key=lambda r: (INTENTS.index(r["intent"]), r["hold"], r["seed"])):
            writer.writerow([r[c] for c in columns])

    status = "PASS" if not failures else "FAIL"
    (out / "summary.json").write_text(json.dumps({
        "P_E": P_E, "P_B": P_B, "num_runs": len(runs),
        "recovery_definition": "(final successes - P_B) / (P_E - P_B), final checkpoint (iteration 5000)",
        "auc_definition": "trapezoidal mean of successes / P_E over the 25 evaluated checkpoints",
        "n80_definition": "first iteration with successes >= 0.8 P_E, interpolated; None if never reached",
        "ceiling_definition": "min(1, (600 / N) / P_E), derivation, not a measurement",
        "test": "Welch t, uncorrected, 3 seeds per group",
        "sources": "grid: train 1111584 + 1116466, eval 1111585, 1116420, 1116422, 1117654, 1117655; N = 1 for hand-2/hand-8/obj-4: M1 train 1104420, eval 1104422/1104730/1108150",
        "failures": failures, "status": status, "groups": summary,
    }, indent=2) + "\n", encoding="utf-8")

    def fmt(x, digits=3):
        return "  -  " if x is None else f"{x:.{digits}g}" if digits < 3 else f"{x:.3f}"

    print(f"{'intent':7} {'N':>3} {'Hz':>5} {'recovery':>14} {'ceil':>5} {'AUC':>13} {'N80':>9} {'fall':>5} "
          f"{'p_vsN1':>7} {'p_vsGoal':>8} {'step1':>6}")
    for e in summary:
        n80 = f"{e['n_reached_80']}/3" + (f" {e['n80_mean_reached']:.0f}" if e["n80_mean_reached"] else "")
        print(f"{e['intent']:7} {e['hold']:3d} {e['rate_hz']:5.1f} {e['recovery_mean']:.3f}±{e['recovery_sd']:.3f} "
              f"{e['ceiling']:5.2f} {e['auc_mean']:.3f}±{e['auc_sd']:.3f} {n80:>9} {e['fall_mean']:5.2f} "
              f"{fmt(e.get('p_recovery_vs_same_intent_n1'), 2):>7} {fmt(e.get('p_recovery_vs_goal4_same_n'), 2):>8} "
              f"{fmt(e['step1_eval_only_recovery_mean']):>6}")

    if not args.skip_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {"hand-2": "#1f77b4", "hand-8": "#d62728", "obj-4": "#2ca02c", "goal-4": "#9467bd"}
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
        rates = np.array([60.0 / h for h in HOLDS])
        for intent in INTENTS:
            es = [e for e in summary if e["intent"] == intent]
            x = [e["rate_hz"] for e in es]
            for axis, key in zip(axes, ["recovery", "auc"]):
                values = [e[f"{key}_values"] if key == "recovery" else [r["auc"] for r in groups[(intent, e["hold"])]] for e in es]
                means = [np.mean(v) for v in values]
                axis.plot(x, means, "o-", color=colors[intent], label=intent)
                for xi, v in zip(x, values):
                    axis.scatter([xi] * len(v), v, color=colors[intent], s=10, alpha=0.4)
        axes[0].plot(rates, [min(1.0, (EPISODE_STEPS / h) / P_E) for h in HOLDS], "k--", lw=1,
                     label="ceiling (1 goal / refresh)")
        axes[0].set(ylabel="Recovery (final checkpoint)", title="Recovery vs intent update rate (trained under hold)")
        axes[1].set(ylabel="AUC (mean successes / P_E)", title="Learning speed (AUC) vs update rate")
        for axis in axes:
            axis.set_xscale("log")
            axis.set_xticks(rates, [f"{r:g}" for r in rates])
            axis.minorticks_off()
            axis.set_xlabel("intent update rate (Hz), N = 1, 6, 12, 30")
            axis.grid(alpha=0.3)
            axis.set_ylim(-0.03, 1.05)
        axes[0].legend(fontsize=8)
        fig.savefig(out / "recovery_vs_rate.png", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(1, len(HOLDS), figsize=(16, 4), sharey=True, constrained_layout=True)
        for axis, hold in zip(axes, HOLDS):
            for intent in INTENTS:
                group = groups.get((intent, hold), [])
                if not group:
                    continue
                grid = group[0]["iterations"]
                stacked = np.vstack([np.interp(grid, r["iterations"], r["values"] / P_E) for r in group])
                axis.plot(grid, stacked.mean(axis=0), color=colors[intent], label=intent)
                axis.fill_between(grid, stacked.min(axis=0), stacked.max(axis=0), color=colors[intent], alpha=0.15)
            axis.axhline(0.8, color="0.5", ls="--", lw=1)
            axis.axhline(min(1.0, (EPISODE_STEPS / hold) / P_E), color="k", ls=":", lw=1)
            axis.set(title=f"N = {hold} ({60 / hold:g} Hz)", xlabel="PPO iteration")
            axis.grid(alpha=0.3)
        axes[0].set_ylabel("successes / P_E")
        axes[0].legend(fontsize=8)
        fig.savefig(out / "learning_curves_by_rate.png", dpi=150)
        plt.close(fig)

    for failure in failures:
        print("check_failure:", failure)
    print(f"rate_sweep_analysis_status={status}")


if __name__ == "__main__":
    main()
