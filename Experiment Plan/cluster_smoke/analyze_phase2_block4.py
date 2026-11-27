#!/usr/bin/env python3
"""Rate sweep phase 2, block 4: Stage 1 (M1) goal-4 and hand-8 x hold N = 1, 6, 30 with 5 seeds (question 4).

Inputs: the 12 block 4 evaluation.json files (seeds 3-4), the phase 1 grid evaluations (seeds 0-2: goal-4 at
N = 1, 6, 30; hand-8 at N = 6, 30) and the M1 grid evaluations (hand-8 N = 1, seeds 0-2).

* Recovery, AUC, N80, ceiling exactly as analyze_rate_sweep.py (M1 constants P_E = 26.11, P_B = 0.006).
* Consistency: the same intent record (task, expert checkpoint and PCA basis hashes, goal hidden, PPO size) for
  seeds 0-4 of a condition; new seeds 3-4 compared with seeds 0-2 (Welch; |z| of each new seed vs old mean/sd).
* Phase 1 contrasts re-tested with 5 seeds (Welch, uncorrected; Mann-Whitney U as a rank check):
  hand-8 vs goal-4 at the same N; each intent vs its own N = 1. Retained = R(N) / R(1) of seed means.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

from analyze_learning_curves import first_crossing, normalized_auc
from analyze_rate_sweep import EPISODE_STEPS, P_B, P_E, load, mean_sd, welch

TASK = "Isaac-Repose-Cube-Shadow-Direct-v0"
INTENTS = ("goal-4", "hand-8")
HOLDS = (1, 6, 30)
OLD, NEW = (0, 1, 2), (3, 4)
RECORD_KEYS = ("task", "expert_checkpoint_sha256", "pca_basis_sha256", "goal_hidden", "asymmetric_critic",
               "num_envs", "max_iterations", "transitions_per_iteration", "intent_hold", "student_obs_dim")


def load_full(path: Path) -> dict:
    r = load(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    r["task"] = report.get("task")
    r["record"] = {k: report["intent_record"].get(k) for k in RECORD_KEYS}
    r["delay"] = int(report.get("intent_delay", 0) or 0)
    r["bits"] = int(report.get("intent_bits", 0) or 0)
    return r


def mwu(a, b):
    return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue) if len(a) > 1 and len(b) > 1 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evals", type=Path, required=True, help="block 4 evaluation copies (seeds 3-4)")
    parser.add_argument("--grid_evals", type=Path, required=True)
    parser.add_argument("--m1_evals", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    wanted = {(i, n) for i in INTENTS for n in HOLDS}
    runs = [load_full(p) for p in sorted(args.evals.glob("*.json"))]
    runs += [r for r in (load_full(p) for p in sorted(args.grid_evals.glob("*.json"))) if (r["intent"], r["hold"]) in wanted]
    runs += [r for r in (load_full(p) for p in sorted(args.m1_evals.glob("*.json")))
             if (r["intent"], r["hold"]) in wanted and "gate" not in r["source"]]

    failures = []
    for r in runs:
        if r["task"] != TASK or r["record"]["task"] != TASK:
            failures.append(f"{r['source']}: task {r['task']} / trained {r['record']['task']}")
        if r["record"]["goal_hidden"] is not True:
            failures.append(f"{r['source']}: goal not hidden")
        if r["hold"] != r["hold_trained"] or r["delay"] or r["bits"]:
            failures.append(f"{r['source']}: eval channel differs from training")
        if r["checkpoints"] != 25 or r["final_iteration"] != 5000:
            failures.append(f"{r['source']}: {r['checkpoints']} checkpoints, final {r['final_iteration']}")
        r["recovery"] = (r["successes"] - P_B) / (P_E - P_B)
        r["auc"] = normalized_auc(r["iterations"], r["values"], P_E)
        r["n80"] = first_crossing(r["iterations"], r["values"], 0.8 * P_E)
        r["ceiling"] = min(1.0, (EPISODE_STEPS / r["hold"]) / P_E)

    groups: dict[tuple[str, int], list[dict]] = {}
    for r in runs:
        groups.setdefault((r["intent"], r["hold"]), []).append(r)
    for key in sorted(wanted):
        g = sorted(groups.get(key, []), key=lambda r: r["seed"])
        groups[key] = g
        if [r["seed"] for r in g] != list(OLD + NEW):
            failures.append(f"{key}: seeds {[r['seed'] for r in g]}")
        # Same training setup across seeds (ignore None fields of old records that lack a key).
        for k in RECORD_KEYS:
            vals = {json.dumps(r["record"][k]) for r in g if r["record"][k] is not None}
            if len(vals) > 1:
                failures.append(f"{key}: intent_record {k} differs across seeds {sorted(vals)}")

    def rec(key, seeds=OLD + NEW):
        return [r["recovery"] for r in groups[key] if r["seed"] in seeds]

    summary = []
    for intent in INTENTS:
        for n in HOLDS:
            key = (intent, n)
            g = groups[key]
            m5, sd5 = mean_sd(rec(key))
            m3, sd3 = mean_sd(rec(key, OLD))
            m_new = mean_sd(rec(key, NEW))[0]
            base5, base3 = mean_sd(rec((intent, 1)))[0], mean_sd(rec((intent, 1), OLD))[0]
            e = {
                "intent": intent, "hold": n, "rate_hz": 60.0 / n,
                "recovery_values": {r["seed"]: r["recovery"] for r in g},
                "recovery_mean_5": m5, "recovery_sd_5": sd5, "recovery_mean_3": m3, "recovery_sd_3": sd3,
                "recovery_mean_new": m_new,
                "new_z": [((v - m3) / sd3) if sd3 else None for v in rec(key, NEW)],
                "p_new_vs_old": welch(rec(key, NEW), rec(key, OLD)),
                "successes_mean_5": float(np.mean([r["successes"] for r in g])),
                "retained_5": m5 / base5, "retained_3": m3 / base3,
                "auc_mean_5": float(np.mean([r["auc"] for r in g])),
                "n80_reached_5": sum(r["n80"] is not None for r in g), "fall_mean_5": float(np.mean([r["fall"] for r in g])),
                "fall_max_5": float(max(r["fall"] for r in g)), "ceiling": g[0]["ceiling"],
            }
            if n != 1:
                e["p_vs_n1_5"] = welch(rec(key), rec((intent, 1)))
                e["p_vs_n1_3"] = welch(rec(key, OLD), rec((intent, 1), OLD))
            if intent == "hand-8":
                other = ("goal-4", n)
                e["p_vs_goal4_5"] = welch(rec(key), rec(other))
                e["p_vs_goal4_3"] = welch(rec(key, OLD), rec(other, OLD))
                e["mwu_vs_goal4_5"] = mwu(rec(key), rec(other))
                e["diff_vs_goal4_5"] = m5 - mean_sd(rec(other))[0]
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

    print(f"P_E={P_E:.3f} P_B={P_B:.4f}")
    for e in summary:
        vals = " ".join(f"{e['recovery_values'][s]:.3f}" for s in OLD + NEW)
        print(f"{e['intent']:6} N={e['hold']:2d} ({e['rate_hz']:4.1f} Hz) R5={f(e['recovery_mean_5'])}±{f(e['recovery_sd_5'])} "
              f"R3={f(e['recovery_mean_3'])} Rnew={f(e['recovery_mean_new'])} [{vals}] new_z={[f(z, 2) for z in e['new_z']]} "
              f"p_new_old={f(e['p_new_vs_old'])} ret5={f(e['retained_5'])} ret3={f(e['retained_3'])} "
              f"fall={e['fall_mean_5']:.3f}(max {e['fall_max_5']:.3f}) n80={e['n80_reached_5']}/5 "
              f"p_n1 5/3={f(e.get('p_vs_n1_5'), 4)}/{f(e.get('p_vs_n1_3'), 4)} "
              f"p_g4 5/3={f(e.get('p_vs_goal4_5'), 4)}/{f(e.get('p_vs_goal4_3'), 4)} mwu5={f(e.get('mwu_vs_goal4_5'), 4)} "
              f"diff={f(e.get('diff_vs_goal4_5'))}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        colors = {"goal-4": "#2a6fdb", "hand-8": "#c0392b"}
        for intent in INTENTS:
            es = [e for e in summary if e["intent"] == intent]
            x = [e["rate_hz"] for e in es]
            ax.errorbar(x, [e["recovery_mean_5"] for e in es], yerr=[e["recovery_sd_5"] for e in es], marker="o",
                        capsize=3, color=colors[intent], label=f"{intent}, 5 seeds")
            for e in es:
                for s, v in e["recovery_values"].items():
                    ax.plot(e["rate_hz"] * (1.04 if intent == "hand-8" else 0.96), v, "o", ms=4, alpha=0.6,
                            color=colors[intent], mfc="white" if s in NEW else colors[intent])
        ax.set_xscale("log")
        ax.set_xticks([60 / n for n in HOLDS], [f"{60 / n:g}" for n in HOLDS])
        ax.set(xlabel="intent update rate (Hz)", ylabel="Recovery (final checkpoint)", ylim=(-0.05, 1.05),
               title="Stage 1, goal hidden: 5 seeds (open = seeds 3-4)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.output_dir / "m1_5seed_recovery_vs_rate.png", dpi=150)
    except ImportError:
        print("matplotlib not available; no figure")

    status = "PASS" if not failures else "FAIL"
    (args.output_dir / "summary.json").write_text(json.dumps({
        "P_E": P_E, "P_B": P_B, "num_runs": len(runs), "status": status, "failures": failures, "groups": summary,
    }, indent=2) + "\n", encoding="utf-8")
    for x in failures:
        print("check_failure:", x)
    print(f"phase2_block4_analysis_status={status}")


if __name__ == "__main__":
    main()
