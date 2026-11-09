#!/usr/bin/env python3
"""Compute N80 and normalized AUC per student run and aggregate them per intent condition.

Inputs are evaluation.json files written by evaluate_students.py. The reference performance P_E is
the expert's metric at its final evaluated checkpoint (an evaluation.json of the expert run).

* N80: first training iteration at which the run's metric reaches 0.8 * P_E, linearly interpolated
  between evaluated checkpoints; reported in iterations and in environment transitions. Runs that
  never reach it are right-censored (reported as None with ``reached_80 = False``).
* AUC: trapezoidal area under metric/P_E over the evaluated iteration range, divided by that range,
  i.e. the mean normalized performance during training (1.0 = expert level throughout).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = {
    "successes": "mean_successes_per_episode",
    "reward": "mean_episode_reward",
}
CONDITION_ORDER = ["blind", "wrist-2", "hand-2", "hand-4", "hand-20", "obj-1", "obj-4", "joint-2", "joint-4"]


def load_curve(path: Path, metric_key: str) -> tuple[dict, np.ndarray, np.ndarray]:
    report = json.loads(path.read_text(encoding="utf-8"))
    points = sorted((entry["iteration"], entry[metric_key]) for entry in report["checkpoints"])
    iterations = np.array([p[0] for p in points], dtype=np.float64)
    values = np.array([p[1] for p in points], dtype=np.float64)
    return report, iterations, values


def first_crossing(iterations: np.ndarray, values: np.ndarray, threshold: float) -> float | None:
    above = np.nonzero(values >= threshold)[0]
    if above.size == 0:
        return None
    index = int(above[0])
    if index == 0:
        return float(iterations[0])
    x0, x1 = iterations[index - 1], iterations[index]
    y0, y1 = values[index - 1], values[index]
    return float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0))


def normalized_auc(iterations: np.ndarray, values: np.ndarray, reference: float) -> float:
    if iterations.size < 2:
        return float(values[0] / reference)
    return float(np.trapezoid(values / reference, iterations) / (iterations[-1] - iterations[0]))


def condition_sort_key(name: str) -> tuple[int, str]:
    return (CONDITION_ORDER.index(name) if name in CONDITION_ORDER else len(CONDITION_ORDER), name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert_eval", required=True, help="evaluation.json of the expert run")
    parser.add_argument("--student_evals", nargs="+", required=True, help="evaluation.json files of student runs")
    parser.add_argument("--metric", choices=sorted(METRICS), default="successes")
    parser.add_argument("--transitions_per_iteration", type=int, default=2048 * 16)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--skip_plot", action="store_true")
    parser.add_argument("--stage_label", default="Stage-1", help="plot title prefix, e.g. M2")
    args = parser.parse_args()

    metric_key = METRICS[args.metric]
    _, expert_iterations, expert_values = load_curve(Path(args.expert_eval), metric_key)
    reference = float(expert_values[-1])
    if reference <= 0:
        raise ValueError(f"Expert reference performance must be positive, got {reference}")
    threshold = 0.8 * reference

    runs = []
    curves = defaultdict(list)
    for path in sorted(args.student_evals):
        report, iterations, values = load_curve(Path(path), metric_key)
        record = report["intent_record"]
        n80 = first_crossing(iterations, values, threshold)
        run = {
            "intent": record["intent"],
            "seed": record.get("seed"),
            "evaluation": str(path),
            "run_dir": report["run_dir"],
            "evaluated_checkpoints": int(iterations.size),
            "final_iteration": float(iterations[-1]),
            "final_value": float(values[-1]),
            "final_normalized": float(values[-1] / reference),
            "reached_80": n80 is not None,
            "n80_iterations": n80,
            "n80_transitions": n80 * args.transitions_per_iteration if n80 is not None else None,
            "auc_normalized": normalized_auc(iterations, values, reference),
        }
        runs.append(run)
        curves[record["intent"]].append((iterations, values))

    conditions = {}
    for intent in sorted({run["intent"] for run in runs}, key=condition_sort_key):
        group = [run for run in runs if run["intent"] == intent]
        reached = [run["n80_iterations"] for run in group if run["reached_80"]]
        aucs = np.array([run["auc_normalized"] for run in group])
        finals = np.array([run["final_normalized"] for run in group])
        conditions[intent] = {
            "seeds": sorted(run["seed"] for run in group),
            "num_runs": len(group),
            "num_reached_80": len(reached),
            "n80_iterations_mean_reached": float(np.mean(reached)) if reached else None,
            "n80_iterations_values": [run["n80_iterations"] for run in group],
            "auc_normalized_mean": float(aucs.mean()),
            "auc_normalized_std": float(aucs.std(ddof=1)) if len(group) > 1 else None,
            "final_normalized_mean": float(finals.mean()),
            "final_normalized_std": float(finals.std(ddof=1)) if len(group) > 1 else None,
        }

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "record_type": "shadow_student_learning_curve_analysis",
        "metric": args.metric,
        "metric_key": metric_key,
        "expert_eval": str(Path(args.expert_eval).resolve()),
        "expert_reference_iteration": float(expert_iterations[-1]),
        "expert_reference_value": reference,
        "n80_threshold": threshold,
        "transitions_per_iteration": args.transitions_per_iteration,
        "n80_definition": "first iteration with metric >= 0.8 * expert reference, linear interpolation between checkpoints; None if never reached",
        "auc_definition": "trapezoidal mean of metric / expert reference over the evaluated iteration range",
        "conditions": conditions,
        "runs": runs,
    }
    (output_dir / f"learning_curves_{args.metric}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output_dir / f"runs_{args.metric}.csv").open("w", newline="", encoding="utf-8") as stream:
        columns = ["intent", "seed", "final_normalized", "reached_80", "n80_iterations", "n80_transitions", "auc_normalized"]
        writer = csv.writer(stream)
        writer.writerow(columns)
        for run in sorted(runs, key=lambda r: (condition_sort_key(r["intent"]), r["seed"] if r["seed"] is not None else -1)):
            writer.writerow([run[column] for column in columns])

    for intent, block in conditions.items():
        n80 = block["n80_iterations_mean_reached"]
        print(
            f"condition {intent:10s} runs={block['num_runs']} reached80={block['num_reached_80']} "
            f"n80_mean={'%.0f' % n80 if n80 is not None else 'none'} "
            f"auc={block['auc_normalized_mean']:.3f} final={block['final_normalized_mean']:.3f}"
        )

    if not args.skip_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        axis.plot(expert_iterations, expert_values / reference, color="black", lw=2, label="expert (seed 0)")
        for index, intent in enumerate(conditions):
            color = colors[index % len(colors)]
            group = curves[intent]
            grid = np.unique(np.concatenate([it for it, _ in group]))
            stacked = np.vstack([np.interp(grid, it, val / reference, left=np.nan, right=np.nan) for it, val in group])
            axis.plot(grid, np.nanmean(stacked, axis=0), color=color, label=f"{intent} (n={len(group)})")
            axis.fill_between(grid, np.nanmin(stacked, axis=0), np.nanmax(stacked, axis=0), color=color, alpha=0.15)
        axis.axhline(0.8, color="0.5", ls="--", lw=1)
        axis.set(xlabel="PPO iteration", ylabel=f"{args.metric} / expert final", title=f"{args.stage_label} learning curves ({args.metric})")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
        fig.savefig(output_dir / f"learning_curves_{args.metric}.png", dpi=150)
        plt.close(fig)
    print("learning_curve_analysis_status=PASS")


if __name__ == "__main__":
    main()
