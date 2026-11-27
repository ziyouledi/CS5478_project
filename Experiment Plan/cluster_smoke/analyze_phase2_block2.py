#!/usr/bin/env python3
"""Rate sweep phase 2, block 2: goal-4 on M1 (Stage 1 task, goal hidden) trained 5000 vs 10000 iterations.

Pairs each resumed run (checkpoints 5200-10000) with its own phase 1 evaluation (200-5000) and reports,
per hold N in {1, 12, 30}: Recovery at 5000 and 10000 (M1 constants, as analyze_rate_sweep.py), the
paired change, and the gap to N = 1. Recovery above the 5000-iteration value means "longer student
training", not "better than the expert" (P_E is the 5000-iteration expert).

Usage: analyze_phase2_block2.py PHASE1_DIR BLOCK2_DIR OUT_DIR
"""
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy import stats

P_E = 26.1103515625
P_B = 0.006184895833333333
EPISODE_STEPS = 600
HOLDS = (1, 12, 30)
RATE = {1: "60 Hz", 12: "5 Hz", 30: "2 Hz"}


def rec(p):
    return (p - P_B) / (P_E - P_B)


def curve(path):
    d = json.loads(Path(path).read_text())
    return d, {c["iteration"]: c["mean_successes_per_episode"] for c in d["checkpoints"]}


def main(phase1_dir, block2_dir, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    runs, errors = [], []
    for f in sorted(Path(block2_dir).glob("*-resume-job*.json")):
        cond = re.match(r"(.+-seed\d)-resume-job", f.name).group(1)
        src = sorted(Path(phase1_dir).glob(f"{cond}-job*.json"))
        d1, c1 = curve(src[0])
        d2, c2 = curve(f)
        if d1["task"] != d2["task"] or d1["intent_hold"] != d2["intent_hold"] or d1["seed"] != d2["seed"]:
            errors.append(f"{f.name}: task/hold/eval seed mismatch")
        if sorted(c1) != list(range(200, 5001, 200)) or sorted(c2) != list(range(5200, 10001, 200)):
            errors.append(f"{f.name}: unexpected checkpoints")
        merged = {**c1, **c2}
        hold = int(d2["intent_hold"])
        runs.append({"run": cond, "hold": hold, "seed": int(re.search(r"seed(\d)", cond).group(1)),
                     "P5000": c1[5000], "P10000": c2[10000],
                     "R5000": rec(c1[5000]), "R10000": rec(c2[10000]),
                     "iterations": sorted(merged), "values": [merged[k] for k in sorted(merged)]})
    table = []
    for h in HOLDS:
        g = sorted([r for r in runs if r["hold"] == h], key=lambda r: r["seed"])
        a = [r["R5000"] for r in g]; b = [r["R10000"] for r in g]
        diff = [y - x for x, y in zip(a, b)]
        table.append({"hold": h, "rate": RATE[h], "n": len(g),
                      "ceiling": min(1.0, (EPISODE_STEPS / h) / P_E),
                      "R5000_mean": statistics.mean(a), "R5000_sd": statistics.stdev(a),
                      "R10000_mean": statistics.mean(b), "R10000_sd": statistics.stdev(b),
                      "delta_mean": statistics.mean(diff),
                      "p_paired": float(stats.ttest_rel(b, a).pvalue),
                      "P5000_mean": statistics.mean(r["P5000"] for r in g),
                      "P10000_mean": statistics.mean(r["P10000"] for r in g)})
    base = {t["hold"]: t for t in table}
    for t in table:
        t["share_of_N1_5000"] = t["R5000_mean"] / base[1]["R5000_mean"]
        t["share_of_N1_10000"] = t["R10000_mean"] / base[1]["R10000_mean"]
        g1 = [r for r in runs if r["hold"] == 1]; gh = [r for r in runs if r["hold"] == t["hold"]]
        if t["hold"] != 1:
            t["p_vs_N1_10000"] = float(stats.ttest_ind([r["R10000"] for r in gh], [r["R10000"] for r in g1], equal_var=False).pvalue)
    status = "PASS" if len(runs) == 9 and not errors else "FAIL"
    summary = {"P_E": P_E, "P_B": P_B, "status": status, "errors": errors,
               "recovery_definition": "(successes - P_B) / (P_E - P_B), M1 constants, single checkpoint",
               "note": "P_E is the 5000-iteration expert; values at 10000 measure longer student training",
               "table": table,
               "runs": [{k: v for k, v in r.items() if k not in ("iterations", "values")} for r in runs]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = {1: "#2a6fdb", 12: "#e08a1e", 30: "#c23b3b"}
    for h in HOLDS:
        g = [r for r in runs if r["hold"] == h]
        it = np.array(g[0]["iterations"])
        v = np.vstack([np.array(r["values"]) for r in g]) / P_E
        m, s = v.mean(0), v.std(0, ddof=1)
        ax.plot(it, m, color=colors[h], label=f"goal-4, {RATE[h]} (N={h})")
        ax.fill_between(it, m - s, m + s, color=colors[h], alpha=0.18, lw=0)
        ax.axhline(min(1.0, (EPISODE_STEPS / h) / P_E), color=colors[h], ls=":", lw=1)
    ax.axvline(5000, color="k", lw=0.8, ls="--"); ax.text(5080, 0.05, "resume", fontsize=8)
    ax.set(xlabel="PPO iteration", ylabel="successes / P_E (expert @5000)",
           title="M1 goal-4: 5000 → 10000 iterations (3 seeds, ±1 sd)")
    ax.legend(fontsize=8, loc="lower right"); fig.tight_layout()
    fig.savefig(out / "m1_goal4_5000_vs_10000.png", dpi=160)

    for t in table:
        print(f"N={t['hold']:>2} ({t['rate']}): R5000={t['R5000_mean']:.3f}±{t['R5000_sd']:.3f}  "
              f"R10000={t['R10000_mean']:.3f}±{t['R10000_sd']:.3f}  delta={t['delta_mean']:+.3f} (paired p={t['p_paired']:.3g})  "
              f"share of N=1: {t['share_of_N1_5000']:.2f} -> {t['share_of_N1_10000']:.2f}  ceiling={t['ceiling']:.2f}"
              + (f"  p_vs_N1@10000={t['p_vs_N1_10000']:.3g}" if 'p_vs_N1_10000' in t else ""))
    for e in errors:
        print("ERROR", e)
    print(f"phase2_block2_analysis_status={status}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
