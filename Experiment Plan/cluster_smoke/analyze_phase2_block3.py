#!/usr/bin/env python3
"""Rate sweep phase 2, block 3: M2 task, goal hidden, hand-8 x hold N = 1, 6, 30 (question 1: does the reversal repeat?).

Inputs: the 9 block 3 evaluation.json files, the 27 block 1 files (blind = P_B, goal-4 / goal-yaw comparators) and,
optionally, the phase 1 summary.json (Stage 1 hand-8 / goal-4 Recovery per N).

* P_E, P_B, Recovery, ceiling, AUC, N80 exactly as in analyze_phase2_block1.py (P_B = mean of the m2-hidden blind runs).
* Because M2's P_B is a goal-free spin strategy at 63% of P_E, conditions are also compared in successes against blind.
* Welch t (uncorrected, 3 seeds): hand-8 vs hand-8 N = 1; vs goal-4 and goal-yaw at the same N; vs blind.
* Retained = R(N) / R(1) of the same intent (seed means), M2 and Stage 1 side by side.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_learning_curves import first_crossing, normalized_auc
from analyze_phase2_block1 import EPISODE_STEPS, P_E, SEEDS, TASK, load, ms, welch

HOLDS = (1, 6, 30)
BLOCK3 = [("hand-8", n) for n in HOLDS]
BLOCK1 = [("blind", 1)] + [(i, n) for i in ("goal-4", "goal-yaw") for n in (1, 6, 12, 30)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evals", type=Path, required=True, help="block 3 evaluation copies")
    parser.add_argument("--block1_evals", type=Path, required=True)
    parser.add_argument("--phase1_summary", type=Path, help="results/rate-sweep-grid-39runs/summary.json")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    runs3 = [load(p) for p in sorted(args.evals.glob("*.json"))]
    runs1 = [load(p) for p in sorted(args.block1_evals.glob("*.json"))]
    failures = []
    for r in runs3 + runs1:
        if r["task"] != TASK or r["train_task"] != TASK:
            failures.append(f"{r['source']}: task {r['task']} / trained {r['train_task']}")
        if r["goal_hidden"] is not True:
            failures.append(f"{r['source']}: goal not hidden")
        if r["hold"] != r["hold_trained"] or r["delay"] or r["bits"]:
            failures.append(f"{r['source']}: eval channel differs from training")
        if r["iterations"].size != 25 or r["iterations"][-1] != 5000:
            failures.append(f"{r['source']}: {r['iterations'].size} checkpoints, last {r['iterations'][-1]}")
    groups: dict[tuple[str, int], list[dict]] = {}
    for r in runs3 + runs1:
        groups.setdefault((r["intent"], r["hold"]), []).append(r)
    for key in BLOCK3 + BLOCK1:
        seeds = sorted(r["seed"] for r in groups.get(key, []))
        if seeds != list(SEEDS):
            failures.append(f"{key}: seeds {seeds}")
    for key in set(groups) - set(BLOCK3 + BLOCK1):
        failures.append(f"unexpected condition {key}")
    if any(r["intent"] != "hand-8" for r in runs3):
        failures.append("block 3 directory contains a non-hand-8 run")

    p_b = float(np.mean([r["successes"] for r in groups.get(("blind", 1), [])] or [np.nan]))
    for r in runs3 + runs1:
        r["recovery"] = (r["successes"] - p_b) / (P_E - p_b)
        r["auc"] = normalized_auc(r["iterations"], r["values"], P_E)
        r["n80"] = first_crossing(r["iterations"], r["values"], 0.8 * P_E)
        r["ceiling"] = min(1.0, (EPISODE_STEPS / r["hold"]) / P_E)

    stage1 = {}
    if args.phase1_summary and args.phase1_summary.exists():
        for g in json.loads(args.phase1_summary.read_text(encoding="utf-8")).get("groups", []):
            if g.get("intent") in ("hand-8", "goal-4"):
                stage1[(g["intent"], int(g["hold"]))] = g.get("recovery_mean")

    def succ(key):
        return [r["successes"] for r in groups.get(key, [])]

    def rec(key):
        return [r["recovery"] for r in groups.get(key, [])]

    blind = succ(("blind", 1))
    summary = []
    for key in BLOCK3 + [k for k in BLOCK1 if k[1] in HOLDS]:
        group = groups.get(key, [])
        if not group:
            continue
        intent, n = key
        rec_m, rec_sd = ms(rec(key))
        s_m, s_sd = ms(succ(key))
        base = ms(rec((intent, 1)))[0] if intent != "blind" else None
        e = {
            "intent": intent, "hold": n, "rate_hz": 60.0 / n, "n_seeds": len(group),
            "successes_mean": s_m, "successes_sd": s_sd,
            "successes_values": [r["successes"] for r in sorted(group, key=lambda r: r["seed"])],
            "recovery_mean": rec_m, "recovery_sd": rec_sd,
            "retained": (rec_m / base) if base else None,
            "auc_mean": ms([r["auc"] for r in group])[0], "ceiling": group[0]["ceiling"],
            "n80_reached": sum(r["n80"] is not None for r in group), "fall_mean": ms([r["fall"] for r in group])[0],
            "fall_max": float(max(r["fall"] for r in group)),
            "p_successes_vs_blind": welch(succ(key), blind) if intent != "blind" else None,
            "stage1_recovery": stage1.get((intent, n)),
        }
        if intent == "hand-8":
            if n != 1:
                e["p_vs_hand8_n1"] = welch(rec(key), rec(("hand-8", 1)))
            e["p_vs_goal4_same_n"] = welch(rec(key), rec(("goal-4", n)))
            e["p_vs_goalyaw_same_n"] = welch(rec(key), rec(("goal-yaw", n)))
            s1 = stage1.get(("hand-8", 1))
            e["stage1_retained"] = (stage1[("hand-8", n)] / s1) if s1 and stage1.get(("hand-8", n)) is not None else None
        summary.append(e)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cols = ["intent", "hold", "seed", "successes", "recovery", "ceiling", "auc", "n80", "fall", "source"]
    with (args.output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        w = csv.writer(stream)
        w.writerow(cols)
        for r in sorted(runs3, key=lambda r: (r["hold"], r["seed"])):
            w.writerow([r.get(c) for c in cols])

    def f(x, d=3):
        return "-" if x is None else f"{x:.{d}f}"

    print(f"P_E={P_E:.3f} P_B={p_b:.3f}")
    for e in summary:
        print(f"{e['intent']:8} N={e['hold']:2d} ({e['rate_hz']:4.1f} Hz) P={e['successes_mean']:6.2f}±{f(e['successes_sd'], 2)} "
              f"{[round(v, 2) for v in e['successes_values']]} rec={f(e['recovery_mean'])}±{f(e['recovery_sd'])} "
              f"ret={f(e['retained'])} fall={e['fall_mean']:.3f} (max {e['fall_max']:.3f}) n80={e['n80_reached']}/3 "
              f"p_blind={f(e['p_successes_vs_blind'])} p_n1={f(e.get('p_vs_hand8_n1'))} "
              f"p_g4={f(e.get('p_vs_goal4_same_n'))} p_gyaw={f(e.get('p_vs_goalyaw_same_n'))} "
              f"stage1_rec={f(e['stage1_recovery'])} stage1_ret={f(e.get('stage1_retained'))}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
        rates = [60 / n for n in HOLDS]
        styles = {"hand-8": "#c0392b", "goal-4": "#2a6fdb", "goal-yaw": "#e08a00"}
        for intent, color in styles.items():
            es = [e for e in summary if e["intent"] == intent]
            axes[0].errorbar([e["rate_hz"] for e in es], [e["successes_mean"] for e in es],
                             yerr=[e["successes_sd"] or 0 for e in es], marker="o", capsize=3, color=color, label=f"M2 {intent}")
        axes[0].axhline(p_b, color="grey", ls=":", label=f"M2 blind (P_B {p_b:.1f})")
        axes[0].axhline(P_E, color="k", ls="-", lw=0.8, label=f"M2 expert (P_E {P_E:.1f})")
        axes[0].plot(rates, [min(P_E, EPISODE_STEPS / n) for n in HOLDS], "k--", lw=1, label="refresh cap")
        axes[0].set_xscale("log")
        axes[0].set_xticks(rates, [f"{r:g}" for r in rates])
        axes[0].set(xlabel="intent update rate (Hz)", ylabel="successes / episode (final checkpoint)",
                    title="M2, goal hidden", ylim=(0, 40))
        axes[0].legend(fontsize=7)
        for task, ls, vals in (("M2", "-", {(e["intent"], e["hold"]): e["recovery_mean"] for e in summary}),
                               ("Stage 1", ":", stage1)):
            for intent in ("hand-8", "goal-4"):
                ys = [vals.get((intent, n)) for n in HOLDS]
                if all(y is not None for y in ys):
                    axes[1].plot(rates, ys, ls, marker="o", color=styles[intent], label=f"{task} {intent}")
        axes[1].axhline(0, color="grey", lw=0.8)
        axes[1].set_xscale("log")
        axes[1].set_xticks(rates, [f"{r:g}" for r in rates])
        axes[1].set(xlabel="intent update rate (Hz)", ylabel="Recovery (task's own P_E, P_B)",
                    title="Reversal: M2 vs Stage 1")
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(args.output_dir / "m2_hand8_vs_rate.png", dpi=150)
    except ImportError:
        print("matplotlib not available; no figure")

    status = "PASS" if not failures else "FAIL"
    (args.output_dir / "summary.json").write_text(json.dumps({
        "P_E": P_E, "P_E_source": "M2 expert ep_5000, job 1032490", "P_B": p_b, "P_B_source": "mean of m2-hidden blind seeds 0-2 (block 1)",
        "num_runs_block3": len(runs3), "status": status, "failures": failures, "groups": summary,
    }, indent=2) + "\n", encoding="utf-8")
    for x in failures:
        print("check_failure:", x)
    print(f"phase2_block3_analysis_status={status}")


if __name__ == "__main__":
    main()
