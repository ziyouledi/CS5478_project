#!/usr/bin/env python3
"""Rate sweep phase 3: students trained under intent delay vs evaluation-only delay vs the waiting-time line.

Inputs
* --phase3_evals: the 12 evaluation.json files of phase 3 (goal-4 at delay 6/30/60, hand-8 at delay 6, seeds 0-2;
  25 checkpoints, evaluated under the trained delay).
* --p1_evals: P1 channel evaluations (60 Hz students, last checkpoint, evaluation-only delay 0/6/30/60/120).
* --grid_evals (optional): rate sweep grid, used for goal-4 and hand-8 trained under hold 6/12/30.

Definitions
* Recovery = (P - P_B) / (P_E - P_B), M1 constants. P = successes per episode at the final checkpoint.
* Time per goal T = 600 / P control steps (60 Hz). T0 = T of the same seed's 60 Hz student without delay (P1 base).
* Extra time = T - T0. Waiting-time line: extra = d for delay d, (N - 1) / 2 for hold N (mean age of the value),
  i.e. Recovery = (600 / (T0 + wait) - P_B) / (P_E - P_B).
* Evaluation-only runs are paired by seed with their own T0. Students trained under delay or hold are different
  networks, so their extra time uses the group-mean T0 of the 60 Hz students.
* Tests: Welch t on Recovery, trained under delay vs evaluation-only at the same delay (3 seeds, uncorrected).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_rate_sweep import EPISODE_STEPS, P_B, P_E, load as load_curve, mean_sd, welch

CONTROL_HZ = 60.0
SEEDS = (0, 1, 2)
PHASE3 = [("goal-4", 6), ("goal-4", 30), ("goal-4", 60), ("hand-8", 6)]
P1_DELAYS = (0, 6, 30, 60, 120)
HOLDS = (6, 12, 30)
LATE = (4000, 5000)  # iterations compared to say whether a curve still rises


def recovery(p: float) -> float:
    return (p - P_B) / (P_E - P_B)


def load_eval(path: Path) -> dict:
    """evaluation.json -> one row; keeps the curve when the evaluation has all checkpoints."""
    report = json.loads(path.read_text(encoding="utf-8"))
    record = report["intent_record"]
    cps = sorted(report["checkpoints"], key=lambda c: c["iteration"])
    last = cps[-1]
    return {
        "intent": record["intent"], "seed": int(record["seed"]), "source": path.name, "task": report["task"],
        "goal_hidden": bool(report.get("goal_hidden")),
        "delay": int(report.get("intent_delay", 0) or 0),
        "delay_trained": int(report.get("intent_delay_trained", record.get("intent_delay", 0)) or 0),
        "hold": int(report.get("intent_hold", 1) or 1),
        "hold_trained": int(report.get("intent_hold_trained", record.get("intent_hold", 1)) or 1),
        "action_hold": int(report.get("action_hold", 1) or 1), "bits": int(report.get("intent_bits", 0) or 0),
        "n_checkpoints": len(cps), "iteration": int(last["iteration"]),
        "curve": {int(c["iteration"]): float(c["mean_successes_per_episode"]) for c in cps},
        "successes": float(last["mean_successes_per_episode"]), "fall": float(last["fall_fraction"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase3_evals", type=Path, required=True)
    parser.add_argument("--p1_evals", type=Path, required=True)
    parser.add_argument("--grid_evals", type=Path, help="rate sweep grid evaluations (trained under hold)")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--skip_plot", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []
    trained = [load_eval(p) for p in sorted(args.phase3_evals.glob("*.json"))]
    p1 = [r for r in (load_eval(p) for p in sorted(args.p1_evals.glob("*.json"))) if r["bits"] == 0]

    for r in trained:
        if r["task"] != "Isaac-Repose-Cube-Shadow-Direct-v0" or not r["goal_hidden"]:
            failures.append(f"{r['source']}: not the Stage 1 task with the goal hidden")
        if r["delay"] != r["delay_trained"] or r["delay"] == 0:
            failures.append(f"{r['source']}: evaluated delay {r['delay']} != trained delay {r['delay_trained']}")
        if r["hold"] != 1 or r["hold_trained"] != 1 or r["action_hold"] != 1 or r["bits"]:
            failures.append(f"{r['source']}: hold / action hold / bits not off")
        if r["n_checkpoints"] != 25 or r["iteration"] != 5000:
            failures.append(f"{r['source']}: {r['n_checkpoints']} checkpoints ending at {r['iteration']}")
    for r in p1:
        if r["delay_trained"] != 0 or r["hold"] != 1 or r["hold_trained"] != 1 or r["iteration"] != 5000:
            failures.append(f"{r['source']}: not a 60 Hz student at ep_5000 without hold")

    tr = {(r["intent"], r["delay"], r["seed"]): r for r in trained}
    ev = {(r["intent"], r["delay"], r["seed"]): r for r in p1}
    if len(tr) != len(trained) or len(ev) != len(p1):
        failures.append("duplicate (intent, delay, seed)")
    for intent, d in PHASE3:
        for s in SEEDS:
            if (intent, d, s) not in tr:
                failures.append(f"missing phase 3 run {(intent, d, s)}")
    for intent in ("goal-4", "hand-8"):
        for d in P1_DELAYS:
            for s in SEEDS:
                if (intent, d, s) not in ev:
                    failures.append(f"missing P1 evaluation {(intent, d, s)}")
    if set(tr) - {(i, d, s) for i, d in PHASE3 for s in SEEDS}:
        failures.append("unexpected phase 3 runs")

    def t0(intent: str, seed: int | None = None) -> float:
        seeds = SEEDS if seed is None else (seed,)
        return EPISODE_STEPS / float(np.mean([ev[(intent, 0, s)]["successes"] for s in seeds]))

    def line(intent: str, wait: float, seed: int | None = None) -> float:
        return recovery(EPISODE_STEPS / (t0(intent, seed) + wait))

    rows = []
    for (intent, d, s), r in sorted(ev.items()):
        rows.append({"intent": intent, "channel": "delay", "value": d, "wait": d, "trained_under": False, "seed": s,
                     "successes": r["successes"], "recovery": recovery(r["successes"]), "fall": r["fall"],
                     "steps_per_goal": EPISODE_STEPS / r["successes"],
                     "extra_steps": EPISODE_STEPS / r["successes"] - t0(intent, s),
                     "line_recovery": line(intent, d, s), "t0": t0(intent, s), "source": r["source"]})
    for (intent, d, s), r in sorted(tr.items()):
        rows.append({"intent": intent, "channel": "delay", "value": d, "wait": d, "trained_under": True, "seed": s,
                     "successes": r["successes"], "recovery": recovery(r["successes"]), "fall": r["fall"],
                     "steps_per_goal": EPISODE_STEPS / r["successes"],
                     "extra_steps": EPISODE_STEPS / r["successes"] - t0(intent),
                     "line_recovery": line(intent, d), "t0": t0(intent),
                     "late_gain": recovery(r["curve"][LATE[1]]) - recovery(r["curve"][LATE[0]]), "source": r["source"]})
    if args.grid_evals:
        for p in sorted(args.grid_evals.glob("*.json")):
            g = load_curve(p)
            if g["intent"] not in ("goal-4", "hand-8") or g["hold"] not in HOLDS or g["seed"] not in SEEDS:
                continue
            if g["hold"] != g["hold_trained"] or g["final_iteration"] != 5000:
                failures.append(f"{g['source']}: hold-trained run not evaluated under its hold at 5000")
            wait = (g["hold"] - 1) / 2
            rows.append({"intent": g["intent"], "channel": "hold", "value": g["hold"], "wait": wait,
                         "trained_under": True, "seed": g["seed"], "successes": g["successes"],
                         "recovery": recovery(g["successes"]), "fall": g["fall"],
                         "steps_per_goal": EPISODE_STEPS / g["successes"],
                         "extra_steps": EPISODE_STEPS / g["successes"] - t0(g["intent"]),
                         "line_recovery": line(g["intent"], wait), "t0": t0(g["intent"]), "source": g["source"]})

    def sel(intent, channel, value, trained_under):
        return sorted((r for r in rows if (r["intent"], r["channel"], r["value"], r["trained_under"])
                       == (intent, channel, value, trained_under)), key=lambda r: r["seed"])

    summary = []
    keys = sorted({(r["intent"], r["channel"], r["value"], r["trained_under"]) for r in rows})
    for intent, channel, value, trained_under in keys:
        g = sel(intent, channel, value, trained_under)
        rec_m, rec_sd = mean_sd([r["recovery"] for r in g])
        ex_m, ex_sd = mean_sd([r["extra_steps"] for r in g])
        e = {"intent": intent, "channel": channel, "value": value, "seconds": value / CONTROL_HZ,
             "trained_under": trained_under, "n_seeds": len(g), "wait_steps": g[0]["wait"],
             "recovery_mean": rec_m, "recovery_sd": rec_sd, "recovery_values": [r["recovery"] for r in g],
             "successes_mean": float(np.mean([r["successes"] for r in g])),
             "line_recovery_mean": float(np.mean([r["line_recovery"] for r in g])),
             "steps_per_goal_mean": float(np.mean([r["steps_per_goal"] for r in g])),
             "extra_steps_mean": ex_m, "extra_steps_sd": ex_sd, "extra_steps_values": [r["extra_steps"] for r in g],
             "fall_mean": float(np.mean([r["fall"] for r in g])), "fall_max": float(max(r["fall"] for r in g))}
        if trained_under and channel == "delay":
            other = sel(intent, "delay", value, False)
            e["eval_only_recovery_mean"] = float(np.mean([r["recovery"] for r in other])) if other else None
            e["p_vs_eval_only"] = welch([r["recovery"] for r in g], [r["recovery"] for r in other])
            e["late_gain_values"] = [r["late_gain"] for r in g]
        summary.append(e)

    # Largest error of the waiting-time line on the evaluation-only delays, per seed (goal-4).
    eo = [r for r in rows if r["intent"] == "goal-4" and not r["trained_under"] and r["value"] > 0]
    model = {"max_abs_recovery_error_eval_only": float(max(abs(r["recovery"] - r["line_recovery"]) for r in eo)),
             "max_abs_wait_error_steps_eval_only": float(max(abs(r["extra_steps"] - r["wait"]) for r in eo)),
             "t0_steps_per_seed": {s: t0("goal-4", s) for s in SEEDS}, "t0_steps_mean": t0("goal-4")}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cols = ["intent", "channel", "value", "trained_under", "seed", "successes", "recovery", "line_recovery",
            "steps_per_goal", "t0", "extra_steps", "wait", "fall", "late_gain", "source"]
    with (args.output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        w = csv.writer(stream)
        w.writerow(cols)
        for r in sorted(rows, key=lambda r: (r["intent"], r["channel"], r["value"], r["trained_under"], r["seed"])):
            w.writerow([r.get(c) for c in cols])

    def f(x, d=3):
        return "-" if x is None else f"{x:.{d}f}"

    print(f"P_E={P_E:.3f} P_B={P_B:.4f} goal-4 T0={model['t0_steps_mean']:.1f} steps "
          f"(per seed {[round(v, 1) for v in model['t0_steps_per_seed'].values()]})")
    print(f"waiting-time line on evaluation-only delay (goal-4, per seed): max |Recovery error| "
          f"{model['max_abs_recovery_error_eval_only']:.3f}, max |wait error| "
          f"{model['max_abs_wait_error_steps_eval_only']:.1f} steps")
    for e in summary:
        tag = "trained" if e["trained_under"] else "eval-only"
        print(f"{e['intent']:6} {e['channel']:5} {e['value']:3d} {tag:9} R={f(e['recovery_mean'])}±{f(e['recovery_sd'])} "
              f"line={f(e['line_recovery_mean'])} extra={f(e['extra_steps_mean'], 1)} (wait {e['wait_steps']:g}) "
              f"fall={f(e['fall_mean'], 2)}"
              + (f" p_vs_eval_only={f(e.get('p_vs_eval_only'), 4)} late_gain={[round(v, 3) for v in e['late_gain_values']]}"
                 if "late_gain_values" in e else ""))

    if not args.skip_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        blue, ink, muted, grid = "#2C67F2", "#17213A", "#62708D", "#D9E0EC"
        plt.rcParams.update({"font.size": 12, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted,
                             "ytick.color": muted, "axes.spines.top": False, "axes.spines.right": False,
                             "axes.grid": True, "grid.color": grid, "grid.linewidth": 0.8, "axes.axisbelow": True})
        fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.3))
        xs = np.linspace(0, 125, 200)

        def series(trained_under, key):
            g = [e for e in summary if e["intent"] == "goal-4" and e["channel"] == "delay"
                 and e["trained_under"] == trained_under]
            return [e["value"] for e in g], [e[key] for e in g], [e[key.replace("_mean", "_values")] for e in g]

        for ax, key, curve in ((a, "recovery_mean", [recovery(EPISODE_STEPS / (t0("goal-4") + x)) for x in xs]),
                               (b, "extra_steps_mean", xs)):
            ax.plot(xs, curve, color=muted, ls="--", lw=1.4, zorder=1)
            x, y, vals = series(False, key)
            ax.plot(x, y, color=blue, lw=0, marker="o", ms=9, mfc="white", mew=2, zorder=3,
                    label="trained at 60 Hz, delay added at evaluation")
            x, y, vals = series(True, key)
            ax.plot(x, y, color=blue, lw=0, marker="o", ms=9, mec="white", mew=1.5, zorder=4,
                    label="trained under the delay")
            for xi, v in zip(x, vals):
                ax.scatter([xi + 3] * len(v), v, s=14, color=ink, zorder=5, linewidths=0)
            ax.set_xticks([0, 30, 60, 90, 120])
            ax.set_xticklabels(["0", "0.5", "1", "1.5", "2"])
            ax.set_xlabel("goal delay (s)")
        a.set_ylabel("Recovery")
        a.set_ylim(0, 0.85)
        a.set_title("goal-4: Recovery under delay", loc="left", fontsize=12, color=ink)
        a.text(84, recovery(EPISODE_STEPS / (t0("goal-4") + 84)) + 0.04, "waiting only", color=muted, fontsize=10)
        a.legend(frameon=False, fontsize=10, loc="upper right")
        b.set_ylabel("extra control steps per goal")
        b.set_ylim(0, 130)
        b.set_title("Time lost per goal (dashed: the delay itself)", loc="left", fontsize=12, color=ink)
        fig.text(0.01, 0.005, "Stage 1 task, goal hidden, delays 0.1 / 0.5 / 1 / 2 s, 3 seeds (small dots), final checkpoint. Extra steps = 600 / successes"
                 " - T0; T0 = steps per goal of the 60 Hz students without delay.", fontsize=8.5, color=muted)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(args.output_dir / "delay_trained_vs_eval_only.png", dpi=150)

    status = "PASS" if not failures else "FAIL"
    (args.output_dir / "summary.json").write_text(json.dumps({
        "P_E": P_E, "P_B": P_B, "recovery_definition": "(P - P_B) / (P_E - P_B)",
        "steps_per_goal_definition": "600 / successes per episode; extra = minus T0 of the 60 Hz students "
                                     "(same seed for evaluation-only rows, group mean for trained rows)",
        "waiting_line": "Recovery = (600 / (T0 + wait) - P_B) / (P_E - P_B); wait = d (delay), (N - 1) / 2 (hold)",
        "status": status, "failures": failures, "waiting_model": model, "groups": summary,
    }, indent=2) + "\n", encoding="utf-8")
    for msg in failures:
        print("check_failure:", msg)
    print(f"phase3_delay_analysis_status={status}")


if __name__ == "__main__":
    main()
