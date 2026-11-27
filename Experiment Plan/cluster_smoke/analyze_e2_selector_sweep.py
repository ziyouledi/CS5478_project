#!/usr/bin/env python3
"""E2' (FACE_TURN_PLAN.md): two-expert task switching with a held / delayed selector message.

Inputs: the per-condition JSONs of one 31_switch_eval.sbatch job run with selector conditions
(``<cycle>[-h<N>|-d<D>|-only-<expert>].json``), and the JSONs of the 60 Hz baseline job 1153366.

Checks and definitions
* Regression: the job's own N = 1, d = 0 runs must equal 1153366 exactly (same checkpoints, seed, envs).
  If not, the sweep is reported but the status is FAIL.
* Waiting per phase w: delay d -> d steps; hold N -> (N - 1) / 2 steps (mean wait for the next refresh,
  refreshes at episode steps that are multiples of N). A cycle of L phases waits L * w.
* Waiting-time line: cycles per 600 steps = 600 / (T0 + L * w), T0 = 600 / cycles of the job's own N = 1 run.
  Measured extra steps per phase = (600 / cycles - T0) / L, compared with w.
* Predicted stale fraction = L * w / (T0 + L * w), compared with the measured fraction of env-steps in which
  the message kind differs from the env's current kind.
Falls and resets are ignored by the line (they are reported per condition).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

STEPS = 600
CONTROL_HZ = 60.0
REGRESSION_KEYS = (
    "phases_completed_per_600_steps", "cycles_completed_per_600_steps", "falls_per_600_steps", "time_outs",
    "fraction_of_steps_by_kind", "by_kind", "by_transition",
)


def load(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    sel = data.get("selector") or {}
    hold, delay, only = int(sel.get("hold", 1)), int(sel.get("delay", 0)), sel.get("only_expert")
    if only:
        channel, wait = f"only-{only}", None
    elif hold > 1:
        channel, wait = f"h{hold}", (hold - 1) / 2.0
    elif delay > 0:
        channel, wait = f"d{delay}", float(delay)
    else:
        channel, wait = "base", 0.0
    return {"file": path.name, "cycle": ",".join(data["cycle"]), "L": len(data["cycle"]), "channel": channel,
            "hold": hold, "delay": delay, "only_expert": only, "wait_steps": wait, "data": data}


def regression(run: dict, base: dict) -> dict:
    diffs = {key: run[key] != base[key] for key in REGRESSION_KEYS}
    rel = abs(run["cycles_completed_per_600_steps"] / base["cycles_completed_per_600_steps"] - 1.0)
    return {"identical": not any(diffs.values()), "differing_keys": [k for k, v in diffs.items() if v],
            "cycles_run": run["cycles_completed_per_600_steps"], "cycles_1153366": base["cycles_completed_per_600_steps"],
            "cycles_relative_diff": rel}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--baseline_dir", type=Path, required=True, help="JSONs of job 1153366")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--skip_plot", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = [load(p) for p in sorted(args.run_dir.glob("*.json"))
            if json.loads(p.read_text(encoding="utf-8")).get("record_type") == "shadow_task_switch_evaluation"]
    cycles = sorted({r["cycle"] for r in rows}, key=lambda c: (len(c), c))
    base = {c: next(r for r in rows if r["cycle"] == c and r["channel"] == "base") for c in cycles}

    checks = {}
    for c in cycles:
        path = args.baseline_dir / (c.replace(",", "-") + ".json")
        checks[c] = regression(base[c]["data"], json.loads(path.read_text(encoding="utf-8")))

    table = []
    for r in rows:
        d, b = r["data"], base[r["cycle"]]["data"]
        t0 = STEPS / b["cycles_completed_per_600_steps"]
        cyc = d["cycles_completed_per_600_steps"]
        entry = {
            "cycle": r["cycle"], "channel": r["channel"], "hold": r["hold"], "delay": r["delay"],
            "wait_steps": r["wait_steps"], "wait_s": None if r["wait_steps"] is None else r["wait_steps"] / CONTROL_HZ,
            "cycles_per_600": cyc, "phases_per_600": d["phases_completed_per_600_steps"],
            "falls_per_600": d["falls_per_600_steps"],
            "cycles_relative_to_base": cyc / b["cycles_completed_per_600_steps"],
            "kind_stale_fraction": (d.get("selector") or {}).get("fraction_env_steps_kind_stale"),
            "fields_replaced_fraction": (d.get("selector") or {}).get("fraction_env_steps_fields_replaced"),
        }
        if r["wait_steps"] is not None:
            lw = r["L"] * r["wait_steps"]
            entry["predicted_cycles_per_600"] = STEPS / (t0 + lw)
            entry["measured_extra_steps_per_phase"] = (STEPS / cyc - t0) / r["L"] if cyc > 0 else None
            entry["predicted_stale_fraction"] = lw / (t0 + lw)
        for kind, k in d["by_kind"].items():
            entry[f"{kind}_median_steps"] = k["steps_to_complete"].get("median")
            entry[f"{kind}_extra_median_steps"] = (
                k["steps_to_complete"].get("median") - b["by_kind"][kind]["steps_to_complete"]["median"]
                if k["steps_to_complete"].get("n") else None)
            entry[f"{kind}_stalled_fraction"] = k["stalled_fraction_of_started"]
            entry[f"{kind}_falls_per_600"] = k["falls_per_600_steps"]
        table.append(entry)
    order = {"base": 0, "h6": 1, "h30": 2, "d6": 3, "d30": 4, "d60": 5}
    table.sort(key=lambda e: (cycles.index(e["cycle"]), order.get(e["channel"], 9), e["channel"]))

    fields = sorted({k for e in table for k in e}, key=lambda k: list(table[0]).index(k) if k in table[0] else 99)
    with (args.output_dir / "e2_selector_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)

    status = "PASS" if all(v["identical"] for v in checks.values()) else "FAIL"
    summary = {"record_type": "e2_selector_sweep_analysis", "run_dir": str(args.run_dir),
               "baseline_dir": str(args.baseline_dir), "regression_vs_1153366": checks, "conditions": table,
               "status": status}
    (args.output_dir / "e2_selector_sweep.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    for c, v in checks.items():
        print(f"regression cycle={c} identical={v['identical']} cycles {v['cycles_run']:.3f} vs {v['cycles_1153366']:.3f}"
              f" differing={v['differing_keys']}")
    print(f"{'cycle':<12}{'chan':<14}{'wait s':>7}{'cycles':>8}{'pred':>7}{'extra/ph':>9}{'stale':>7}{'pred':>7}{'falls':>7}")
    for e in table:
        def fmt(x, spec):
            return format(x, spec) if x is not None else format("-", ">" + spec.split(".")[0])
        print(f"{e['cycle']:<12}{e['channel']:<14}{fmt(e['wait_s'], '7.2f')}{e['cycles_per_600']:8.3f}"
              f"{fmt(e.get('predicted_cycles_per_600'), '7.2f')}{fmt(e.get('measured_extra_steps_per_phase'), '9.1f')}"
              f"{fmt(e['kind_stale_fraction'], '7.3f')}{fmt(e.get('predicted_stale_fraction'), '7.3f')}"
              f"{e['falls_per_600']:7.3f}")

    if not args.skip_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        ink, muted, grid = "#17213A", "#62708D", "#D9E0EC"
        colors = {"up,turn": "#2C67F2", "any,up,turn": "#D9622B"}
        plt.rcParams.update({"font.size": 12, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted,
                             "ytick.color": muted, "axes.spines.top": False, "axes.spines.right": False,
                             "axes.grid": True, "grid.color": grid, "grid.linewidth": 0.8, "axes.axisbelow": True})
        fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.3))
        swept = [e for e in table if e["wait_steps"] is not None]
        wmax = max(e["wait_steps"] for e in swept)
        xs = np.linspace(0, wmax * 1.05, 200)
        for c in cycles:
            color, L = colors.get(c, ink), base[c]["L"]
            t0 = STEPS / base[c]["data"]["cycles_completed_per_600_steps"]
            a.plot(xs / CONTROL_HZ, STEPS / (t0 + L * xs), color=color, ls="--", lw=1.4, zorder=1)
            for kind, marker_face in (("h", "white"), ("d", color)):
                pts = [e for e in swept if e["cycle"] == c and (e["channel"].startswith(kind) or e["channel"] == "base")]
                pts = [e for e in pts if e["channel"] != "base" or kind == "d"]
                a.plot([e["wait_s"] for e in pts], [e["cycles_per_600"] for e in pts], lw=0, marker="o", ms=8,
                       color=color, mfc=marker_face, mew=2, zorder=3)
                ex = [e for e in pts if e["channel"] != "base" and e.get("measured_extra_steps_per_phase") is not None]
                b.plot([e["wait_steps"] for e in ex], [e["measured_extra_steps_per_phase"] for e in ex], lw=0,
                       marker="o", ms=8, color=color, mfc=marker_face, mew=2, zorder=3)
            a.text(xs[-1] / CONTROL_HZ, STEPS / (t0 + L * xs[-1]), f"  {c}", color=ink, fontsize=10, va="center")
        b.plot(xs, xs, color=muted, ls="--", lw=1.4, zorder=1)
        a.set_xlabel("mean wait for a new message per phase (s)")
        a.set_ylabel("cycles per 600 steps")
        a.set_title("Completed cycles (dashed: waiting only)", loc="left", fontsize=12, color=ink)
        b.set_xlabel("mean wait per phase (steps)")
        b.set_ylabel("extra steps per phase")
        b.set_title("Time lost per phase (dashed: the wait itself)", loc="left", fontsize=12, color=ink)
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], lw=0, marker="o", ms=8, color=colors[c], label=c) for c in cycles if c in colors]
        handles += [Line2D([], [], lw=0, marker="o", ms=8, color=muted, mfc="white", mew=2, label="hold N (wait (N-1)/2)"),
                    Line2D([], [], lw=0, marker="o", ms=8, color=muted, label="delay d (wait d)")]
        b.legend(handles=handles, frameon=False, fontsize=9, loc="upper left")
        a.set_ylim(bottom=0)
        b.set_ylim(bottom=min(0, *(e["measured_extra_steps_per_phase"] for e in swept
                                   if e.get("measured_extra_steps_per_phase") is not None)))
        fig.tight_layout()
        fig.savefig(args.output_dir / "e2_selector_sweep.png", dpi=150)

    print(f"e2_selector_sweep_analysis_status={status}")


if __name__ == "__main__":
    main()
