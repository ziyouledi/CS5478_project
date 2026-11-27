#!/usr/bin/env python3
"""Rate sweep phase 4: slow expert (Stage 1 task, goal visible, no intent, action applied every N steps).

Inputs
* --phase4_evals: 9 evaluation.json files (blind, action hold 6 / 12 / 30, seeds 0-2; 25 checkpoints,
  evaluated under the trained action hold).
* --stage1_evals: Stage 1 blind students (action hold 1, seeds 0-2): the 60 Hz reference.
* --grid_evals (optional): rate sweep grid (goal hidden, intent trained under hold N) for hand-8 and goal-4
  at the same rates, plus --m1_evals for hand-8 at N = 1.

Definitions
* P = successes per episode at the final checkpoint; P / P_E with the expert P_E. Goal-hidden students are
  reported as Recovery = (P - P_B) / (P_E - P_B); P_B is 0.006, so the two scales differ by < 0.001.
* AUC = trapezoidal mean of P / P_E over the checkpoints. late gain = P(5000) - P(4000), in successes.
* Tests: Welch t on P (3 seeds, uncorrected): slow expert vs hand-8 and vs goal-4 trained at the same rate.
* Lower bound: the policy is queried every step and PPO also updates on the unapplied actions (no frame skip).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_learning_curves import normalized_auc
from analyze_rate_sweep import P_B, P_E, load as load_curve, mean_sd, welch

SEEDS = (0, 1, 2)
HOLDS = (1, 6, 12, 30)
LATE = (4000, 5000)


def load_eval(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    record = report["intent_record"]
    cps = sorted(report["checkpoints"], key=lambda c: c["iteration"])
    its = np.array([c["iteration"] for c in cps], dtype=np.float64)
    vals = np.array([c["mean_successes_per_episode"] for c in cps], dtype=np.float64)
    last = cps[-1]
    return {
        "intent": record["intent"], "seed": int(record["seed"]), "source": path.name, "task": report["task"],
        "goal_hidden": bool(report.get("goal_hidden", record.get("goal_hidden", False))),
        "action_hold": int(report.get("action_hold", 1) or 1),
        "action_hold_trained": int(record.get("action_hold", 1) or 1),
        "intent_hold": int(report.get("intent_hold", 1) or 1), "delay": int(report.get("intent_delay", 0) or 0),
        "iterations": its, "values": vals, "n_checkpoints": len(cps), "iteration": int(last["iteration"]),
        "successes": float(vals[-1]), "fall": float(last["fall_fraction"]),
        "any_success": float(last["fraction_episodes_with_success"]),
        "episode_length": float(last["mean_episode_length"]),
        "auc": normalized_auc(its, vals, P_E),
        "late_gain": float(vals[list(its).index(LATE[1])] - vals[list(its).index(LATE[0])]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase4_evals", type=Path, required=True)
    parser.add_argument("--stage1_evals", type=Path, required=True)
    parser.add_argument("--grid_evals", type=Path)
    parser.add_argument("--m1_evals", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--skip_plot", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []
    slow = [load_eval(p) for p in sorted(args.phase4_evals.glob("*.json"))]
    ref = [load_eval(p) for p in sorted(args.stage1_evals.glob("blind-seed*.json"))]
    for r in slow + ref:
        if r["task"] != "Isaac-Repose-Cube-Shadow-Direct-v0" or r["intent"] != "blind" or r["goal_hidden"]:
            failures.append(f"{r['source']}: not Stage 1, blind, goal visible")
        if r["action_hold"] != r["action_hold_trained"] or r["intent_hold"] != 1 or r["delay"]:
            failures.append(f"{r['source']}: evaluated action hold != trained, or intent hold / delay set")
        if r["n_checkpoints"] != 25 or r["iteration"] != 5000:
            failures.append(f"{r['source']}: {r['n_checkpoints']} checkpoints ending at {r['iteration']}")
    for r in ref:
        if r["action_hold"] != 1:
            failures.append(f"{r['source']}: reference with action hold")
    runs = {(r["action_hold"], r["seed"]): r for r in slow + ref}
    if len(runs) != len(slow) + len(ref):
        failures.append("duplicate (action hold, seed)")
    for n in HOLDS:
        for s in SEEDS:
            if (n, s) not in runs:
                failures.append(f"missing action hold {n} seed {s}")
    if set(runs) - {(n, s) for n in HOLDS for s in SEEDS}:
        failures.append("unexpected runs")

    # Goal-hidden students trained under an intent hold at the same rate (their own loop runs at 60 Hz).
    students: dict[tuple[str, int], list[float]] = {}
    for d in (args.grid_evals, args.m1_evals):
        if not d:
            continue
        for p in sorted(d.glob("*.json")):
            if "gate" in p.name:
                continue
            g = load_curve(p)
            if g["intent"] in ("hand-8", "goal-4") and g["hold"] in HOLDS and g["seed"] in SEEDS \
                    and g["hold"] == g["hold_trained"]:
                students.setdefault((g["intent"], g["hold"]), []).append(g["successes"])
    for key, v in students.items():
        if len(v) != 3:
            failures.append(f"student group {key}: {len(v)} runs")

    summary = []
    for n in HOLDS:
        g = [runs[(n, s)] for s in SEEDS if (n, s) in runs]
        if not g:
            continue
        p_m, p_sd = mean_sd([r["successes"] for r in g])
        e = {"action_hold": n, "rate_hz": 60.0 / n, "n_seeds": len(g),
             "successes_values": [r["successes"] for r in g], "successes_mean": p_m, "successes_sd": p_sd,
             "ratio_values": [r["successes"] / P_E for r in g], "ratio_mean": p_m / P_E,
             "ratio_sd": None if p_sd is None else p_sd / P_E,
             "auc_mean": float(np.mean([r["auc"] for r in g])),
             "fall_values": [r["fall"] for r in g], "fall_mean": float(np.mean([r["fall"] for r in g])),
             "any_success_mean": float(np.mean([r["any_success"] for r in g])),
             "late_gain_values": [r["late_gain"] for r in g],
             "late_gain_relative": [r["late_gain"] / r["successes"] if r["successes"] > 0.5 else None for r in g]}
        for intent in ("hand-8", "goal-4"):
            v = students.get((intent, n))
            if v:
                tag = intent.replace("-", "")
                e[f"{tag}_recovery_mean"] = float(np.mean([(x - P_B) / (P_E - P_B) for x in v]))
                e[f"p_vs_{tag}"] = welch([r["successes"] for r in g], v)
        summary.append(e)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cols = ["action_hold", "seed", "successes", "ratio", "auc", "fall", "any_success", "episode_length",
            "late_gain", "source"]
    with (args.output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        w = csv.writer(stream)
        w.writerow(cols)
        for (n, s), r in sorted(runs.items()):
            w.writerow([n, s, r["successes"], r["successes"] / P_E, r["auc"], r["fall"], r["any_success"],
                        r["episode_length"], r["late_gain"], r["source"]])

    def f(x, d=3):
        return "-" if x is None else f"{x:.{d}f}"

    print(f"P_E={P_E:.3f}")
    for e in summary:
        print(f"N={e['action_hold']:2d} ({e['rate_hz']:4.1f} Hz) P/P_E={f(e['ratio_mean'])}±{f(e['ratio_sd'])} "
              f"{[round(v, 3) for v in e['ratio_values']]} AUC={f(e['auc_mean'])} fall={f(e['fall_mean'], 2)} "
              f"any_success={f(e['any_success_mean'], 2)} late_gain={[round(v, 2) for v in e['late_gain_values']]} "
              f"hand-8={f(e.get('hand8_recovery_mean'))} (p={f(e.get('p_vs_hand8'), 4)}) "
              f"goal-4={f(e.get('goal4_recovery_mean'))} (p={f(e.get('p_vs_goal4'), 4)})")

    if not args.skip_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ink, muted, grid = "#17213A", "#62708D", "#D9E0EC"
        col = {"slow": "#17213A", "hand-8": "#D9602F", "goal-4": "#2C67F2"}
        plt.rcParams.update({"font.size": 12, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted,
                             "ytick.color": muted, "axes.spines.top": False, "axes.spines.right": False,
                             "axes.grid": True, "grid.color": grid, "grid.linewidth": 0.8, "axes.axisbelow": True})
        fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.3))
        x = list(range(len(HOLDS)))
        labels = [f"{60 // n} Hz" for n in HOLDS]
        for intent in ("goal-4", "hand-8"):
            y = [np.mean([(v - P_B) / (P_E - P_B) for v in students[(intent, n)]]) if (intent, n) in students else np.nan
                 for n in HOLDS]
            a.plot(x, y, color=col[intent], lw=2, marker="o", ms=8, mec="white", mew=1.5)
            a.text(x[-1] + 0.12, y[-1] + (0.0 if intent == "goal-4" else 0.07), f"{intent} student\n(goal hidden, acts at 60 Hz)",
                   color=col[intent], fontsize=9, va="center")
        y = [e["ratio_mean"] for e in summary]
        a.plot(x, y, color=col["slow"], lw=2.4, marker="s", ms=8, mec="white", mew=1.5)
        for xi, e in zip(x, summary):
            a.scatter([xi + 0.07] * len(e["ratio_values"]), e["ratio_values"], s=14, color=muted, zorder=5, linewidths=0)
        a.text(x[-1] + 0.12, y[-1] - 0.05, "slow expert\n(sees the goal)", color=col["slow"], fontsize=9, va="center")
        a.set_xticks(x)
        a.set_xticklabels(labels)
        a.set_xlim(-0.3, 4.6)
        a.set_ylim(-0.1, 1.05)
        a.set_xlabel("rate of the slow part (action or intent)")
        a.set_ylabel("successes / expert")
        a.set_title("Acting slowly vs being told slowly", loc="left", fontsize=12, color=ink)
        for n, shade in zip(HOLDS, ("#9AA5BD", "#17213A", "#62708D", "#B8C0D2")):
            for s in SEEDS:
                r = runs.get((n, s))
                if r is not None:
                    b.plot(r["iterations"], r["values"] / P_E, color=shade, lw=1.6 if n == 6 else 1.1)
            ends = [runs[(n, s)]["successes"] / P_E for s in SEEDS if (n, s) in runs]
            if ends:
                b.text(5080, float(np.exp(np.mean(np.log(ends)))), f"{60 // n} Hz", color=ink, fontsize=10, va="center")
        b.set_yscale("log")
        b.set_ylim(1e-4, 1.5)
        b.set_xlabel("PPO iteration")
        b.set_ylabel("successes / expert (log)")
        b.set_title("Slow expert: learning curves by action rate", loc="left", fontsize=12, color=ink)
        b.set_xlim(0, 5700)
        fig.text(0.01, 0.005, "Stage 1 task. Slow expert: goal visible, a new action is applied every N control steps and "
                 "held between (lower bound: no frame skip). Small dots: seeds.", fontsize=8.5, color=muted)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(args.output_dir / "slow_expert_vs_rate.png", dpi=150)

    status = "PASS" if not failures else "FAIL"
    (args.output_dir / "summary.json").write_text(json.dumps({
        "P_E": P_E, "P_B": P_B, "ratio_definition": "successes per episode / P_E (final checkpoint)",
        "caveat": "lower bound: policy queried every step, PPO updates on unapplied actions, no phase in the observation",
        "status": status, "failures": failures, "groups": summary,
    }, indent=2) + "\n", encoding="utf-8")
    for msg in failures:
        print("check_failure:", msg)
    print(f"phase4_action_hold_analysis_status={status}")


if __name__ == "__main__":
    main()
