#!/usr/bin/env python3
"""Figures for the final presentation, from the result summaries in Experiment Plan/results/.

Palette (validated with the dataviz validator, light surface; all checks PASS):
goal-4 #2C67F2, hand-8 #D9602F, hand-2 #8A68E8, obj-4 #1E9C77. Colour follows the intent in every figure;
goal-yaw = goal-4 hue, dashed, open markers. Every line is also direct-labelled.
Block 3 (M2 hand-8) is optional: M2 figure draws whatever block 3 evaluations exist in
Docs/Milestone/data/phase2_block3_evals/.
"""
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

PCT = PercentFormatter(1.0, decimals=0)  # recovery is a 0-1 fraction; show it as a percentage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = ROOT / "Experiment Plan" / "results"
DATA = ROOT / "Docs" / "Milestone" / "data"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

C = {"goal-4": "#2C67F2", "hand-8": "#D9602F", "hand-2": "#8A68E8", "obj-4": "#1E9C77", "goal-yaw": "#2C67F2"}
INK, MUTED, GRID = "#17213A", "#62708D", "#D9E0EC"
RATES = {1: 60, 6: 10, 12: 5, 30: 2}

plt.rcParams.update({
    "font.size": 13, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", OUT / name)


def fig_m1_recovery():
    d = json.loads((RES / "m1-curves-27runs" / "stats_vs_blind.json").read_text())["conditions"]
    pe, pb = 26.1103515625, 0.006184895833333333
    order = sorted(d, key=lambda k: d[k]["recovery_mean"])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for i, k in enumerate(order):
        vals = [(f - pb) / (pe - pb) for f in d[k]["final"]]
        col = C[k] if k in ("hand-8", "hand-2", "obj-4") else "#AEB8CC"
        ax.barh(i, d[k]["recovery_mean"], height=0.62, color=col, alpha=0.9)
        ax.scatter(vals, [i] * 3, s=22, color=INK, zorder=3, linewidths=0)
        ax.text(1.04, i, f"{d[k]['recovery_mean']:.0%}", va="center", fontsize=11, color=INK)
    ax.set_yticks(range(len(order)), [k.replace("wrist2-finger-2", "wrist-2+finger-2") for k in order])
    ax.set_xlim(0, 1.13)
    ax.axvline(1, color=MUTED, ls="--", lw=1)
    ax.text(1.0, len(order) - 0.4, "expert", ha="center", fontsize=10, color=MUTED)
    ax.set_xlabel("Recovery  (0% = blind, 100% = expert)"); ax.xaxis.set_major_formatter(PCT)
    ax.grid(axis="y", visible=False)
    ax.set_title("bar = mean of 3 seeds, dots = seeds", fontsize=11, color=MUTED, loc="left")
    save(fig, "m1_recovery.png")


def fig_rate_sweep():
    g = json.loads((RES / "rate-sweep-grid-39runs" / "summary.json").read_text())["groups"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    xs = [60, 10, 5, 2]
    for intent in ("goal-4", "hand-8", "hand-2", "obj-4"):
        rows = sorted([r for r in g if r["intent"] == intent], key=lambda r: -r["rate_hz"])
        x = [r["rate_hz"] for r in rows]; y = [r["recovery_mean"] for r in rows]; e = [r["recovery_sd"] for r in rows]
        ax.errorbar(x, y, yerr=e, color=C[intent], lw=2.2, marker="o", ms=8, capsize=3,
                    markeredgecolor="white", markeredgewidth=1.5, label=intent)
        ax.text(x[0] * 1.13, y[0], intent, color=INK, fontsize=11, va="center", ha="right")
    ceil = [min(1, (600 / n) / 26.1103515625) for n in (1, 6, 12, 30)]
    ax.plot(xs, ceil, color=MUTED, ls=":", lw=1.4)
    ax.text(2.1, ceil[-1] + 0.04, "refresh ceiling", color=MUTED, fontsize=10)
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set_xticks(xs, ["60 Hz\n(every step)", "10 Hz", "5 Hz", "2 Hz"]); ax.minorticks_off()
    ax.set_xlim(130, 1.45); ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Recovery"); ax.yaxis.set_major_formatter(PCT); ax.set_xlabel("intent update rate (trained under the same rate)")
    ax.legend(loc="upper right", ncol=4, fontsize=10, bbox_to_anchor=(1, 1.12))
    save(fig, "rate_sweep.png")


def fig_channel():
    g = json.loads((RES / "rate-sweep-p1-channel" / "summary.json").read_text())["groups"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True, gridspec_kw={"width_ratios": [1.5, 1]})
    for intent in ("goal-4", "hand-8"):
        rows = sorted([r for r in g if r["intent"] == intent and r["kind"] == "delay"], key=lambda r: r["delay_s"])
        x = [r["delay_s"] for r in rows]; y = [r["recovery_mean"] for r in rows]; e = [r["recovery_std"] for r in rows]
        a.errorbar(x, y, yerr=e, color=C[intent], lw=2.2, marker="o", ms=8, capsize=3, markeredgecolor="white", markeredgewidth=1.5, label=intent)
        a.text(x[-1] + 0.05, y[-1] + 0.03, intent, color=INK, fontsize=11, ha="right" if intent == "hand-8" else "left")
    a.set_xlabel("delay (s)"); a.set_ylabel("Recovery"); a.set_title("Latency", loc="left", fontsize=12, color=INK)
    a.set_ylim(-0.05, 1.0); a.yaxis.set_major_formatter(PCT); a.legend(loc="upper right", fontsize=10)
    rows = [r for r in g if r["intent"] == "goal-4" and r["kind"] == "bits"]
    base = [r for r in g if r["intent"] == "goal-4" and r["kind"] == "delay" and r["delay_s"] == 0][0]
    labels = ["float"] + [str(int(r["value"])) for r in sorted(rows, key=lambda r: -r["value"])]
    vals = [base["recovery_mean"]] + [r["recovery_mean"] for r in sorted(rows, key=lambda r: -r["value"])]
    b.bar(range(len(vals)), vals, color=C["goal-4"], width=0.62)
    for i, v in enumerate(vals):
        b.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=10, color=INK)
    b.set_xticks(range(len(vals)), labels); b.set_xlabel("bits per goal-4 component")
    b.set_title("Quantisation (goal-4)", loc="left", fontsize=12, color=INK); b.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "channel_delay_bits.png")


def fig_training_length():
    s = json.loads((RES / "phase2-block2-resume" / "summary.json").read_text())
    t = {r["hold"]: r for r in s["table"]}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = [60, 5, 2]
    for key, col, lab in (("R5000_mean", "#9DB7F5", "5000 iterations"), ("R10000_mean", C["goal-4"], "10000 iterations")):
        off = -0.18 if key == "R5000_mean" else 0.18
        ys = [t[h][key] for h in (1, 12, 30)]
        ax.bar([i + off for i in range(3)], ys, width=0.34, color=col, label=lab)
        for i, v in enumerate(ys):
            ax.text(i + off, v + 0.015, f"{v:.0%}", ha="center", fontsize=10, color=INK)
    ax.set_xticks(range(3), [f"{r} Hz" for r in x]); ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recovery (goal-4)"); ax.yaxis.set_major_formatter(PCT); ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title("Longer training lifts fast rates; 2 Hz stays put", loc="left", fontsize=12, color=INK)
    save(fig, "training_length.png")


def fig_m2():
    g = json.loads((RES / "phase2-block1-m2" / "summary.json").read_text())["groups"]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    blind = [r for r in g if r["intent"] == "blind"][0]["successes_mean"]
    pe = 31.137
    ax.axhline(pe, color=MUTED, ls="--", lw=1); ax.text(1.6, pe + 0.6, "M2 expert", color=MUTED, fontsize=10, ha="right")
    ax.axhline(blind, color="#9AA6BF", lw=2); ax.text(70, blind - 1.8, "blind (spins the cube)", color=MUTED, fontsize=10, ha="left")
    for intent, ls, mfc in (("goal-4", "-", None), ("goal-yaw", "--", "white")):
        rows = sorted([r for r in g if r["intent"] == intent], key=lambda r: -r["rate_hz"])
        x = [r["rate_hz"] for r in rows]; y = [r["successes_mean"] for r in rows]
        ax.plot(x, y, color=C[intent], ls=ls, lw=2.2, marker="o", ms=8, mfc=mfc or C[intent], markeredgewidth=1.5, label=intent)
    # Block 3 (hand-8 on M2), whatever is available.
    b3 = DATA / "phase2_block3_evals"
    if b3.exists():
        by = {}
        for f in b3.glob("*.json"):
            d = json.loads(f.read_text())
            last = max(d["checkpoints"], key=lambda c: c["iteration"])
            by.setdefault(int(d["intent_hold"]), []).append(last["mean_successes_per_episode"])
        if by:
            hs = sorted(by)
            ax.plot([RATES[h] for h in hs], [statistics.mean(by[h]) for h in hs], color=C["hand-8"], lw=2.2, marker="o",
                    ms=8, markeredgecolor="white", markeredgewidth=1.5,
                    label="hand-8" + ("" if 12 in by else " (no 5 Hz run)"))
    ax.set_xscale("log"); ax.invert_xaxis(); ax.set_xticks([60, 10, 5, 2], ["60 Hz", "10 Hz", "5 Hz", "2 Hz"]); ax.minorticks_off()
    ax.set_xlim(75, 1.45); ax.set_ylim(0, 40)
    ax.set_ylabel("successes per episode")
    ax.legend(loc="lower left", fontsize=10)
    ax.set_title("M2 (rotate about z), goal hidden", loc="left", fontsize=12, color=INK)
    save(fig, "m2_rate.png")


def fig_staleness(keys=("rate", "err", "tol"), name="staleness.png"):
    """Retained Recovery (vs the same intent at 60 Hz) against three candidate predictors."""
    rows = list(csv.DictReader((RES / "rate-sweep-staleness" / "points.csv").open()))
    s = json.loads((RES / "rate-sweep-staleness" / "summary.json").read_text())["predictors"]
    rho = {"rate": s["retained~log2_hold"]["spearman"], "err": s["retained~normalized_hold_error"]["spearman"],
           "tol": s["retained~stale_tol (obj-4, goal-4 only)"]["spearman"]}
    panels = [("rate", lambda r: 60 / float(r["hold"]), "update rate (Hz)",
               f"A  Rate: one curve per intent  (|ρ| = {abs(rho['rate']):.2f})"),
              ("err", lambda r: float(r["normalized_hold_error"]), "how wrong the held value is\n(error of the held intent / its variance, expert rollout)",
               f"B  Staleness: one trend  (|ρ| = {abs(rho['err']):.2f})"),
              ("tol", lambda r: float(r["stale_tol"]) if r["stale_tol"] else None, "share of time the held value is\noutside the 0.1 rad success tolerance",
               f"C  Against the tolerance  (|ρ| = {abs(rho['tol']):.2f}, n = 6)")]
    LAB = {"err": {"goal-4": (0.05, 0.03, "left"), "obj-4": (-0.02, -0.07, "center"),
                   "hand-8": (0.0, 0.26, "right"), "hand-2": (-0.42, 0.03, "right")},
           "tol": {"goal-4": (-0.03, -0.06, "right"), "obj-4": (0.03, 0.0, "left")}}
    panels = [q for q in panels if q[0] in keys]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.8 * len(panels), 4.7), sharey=True)
    for ax, (key, fx, xl, title) in zip(axes, panels):
        for intent in ("goal-4", "obj-4", "hand-8", "hand-2"):
            rr = sorted([r for r in rows if r["intent"] == intent and fx(r) is not None], key=lambda r: int(r["hold"]))
            if not rr:
                continue
            x = [fx(r) for r in rr]; y = [float(r["retained"]) for r in rr]
            ax.plot(x, y, color=C[intent], lw=2.2 if key == "rate" else 1.4, alpha=1 if key == "rate" else 0.45)
            ax.plot(x, y, color=C[intent], ls="none", marker="o", ms=8, markeredgecolor="white", markeredgewidth=1.5)
            if key == "rate":
                ly = {"goal-4": y[-1], "obj-4": 0.125, "hand-8": 0.06, "hand-2": -0.005}[intent]
                ax.text(x[-1] * 0.88, ly, intent, color=INK, fontsize=11, va="center", ha="left")
            else:
                dx, dy, ha = LAB[key][intent]
                ax.text(x[-1] + dx, y[-1] + dy, intent, color=INK, fontsize=11, va="center", ha=ha)
        if key == "rate":
            ax.set_xscale("log"); ax.invert_xaxis(); ax.set_xlim(90, 0.95)
            ax.set_xticks([60, 10, 5, 2], ["60", "10", "5", "2"]); ax.minorticks_off()
        if key == "err":
            ax.plot([0.49, 0.49], [0.44, 0.80], color=INK, lw=1.2, ls=":")
            ax.text(0.03, 0.17, "obj-4 and goal-4:\nsame staleness,\ndifferent loss", fontsize=10.5, color=MUTED, va="center")
            ax.set_xlim(-0.08, 2.15)
            ax.set_xticks([0, 0.5, 1, 1.5, 2], ["0\nexact", "0.5", "1\nno better than\na constant", "1.5", "2\nunrelated"])
        if key == "tol":
            ax.set_xlim(-0.04, 1.12); ax.xaxis.set_major_formatter(PCT)
        ax.set_ylim(-0.05, 1.08)
        ax.set_xlabel(xl, fontsize=11.5)
        ax.set_title(title, fontsize=11.5, color=INK, loc="left")
    axes[0].yaxis.set_major_formatter(PCT)
    axes[0].set_ylabel("Recovery retained\n(relative to the same intent at 60 Hz)")
    fig.tight_layout()
    save(fig, name)


if __name__ == "__main__":
    fig_m1_recovery(); fig_rate_sweep(); fig_channel(); fig_training_length(); fig_m2(); fig_staleness(); fig_staleness(("rate", "err"), "staleness_ab.png")
