#!/usr/bin/env python3
"""E3 (FACE_TURN_PLAN.md): goal-hidden students of the frozen v4 unified teacher, evaluated on the unified task.

Inputs (local copies of 31_switch_eval.sbatch outputs, one directory per array task with the per-condition JSONs):
* --grid_dir   evaluations 1160763_0-14 of training 1160744_0-14 (task<i>/{random,any,up,up-turn,any-up-turn}.json),
               last checkpoint (iteration 5000) of each run.
* --diag_dir   goal-7 diagnostic 1163578_0-2 (task<t>/{any,up}-it<N>.json): goal-7 hold 1 seeds 0-2 at iterations
               1000-5000, with the goal7_intent_check (intent the actor received vs the env's current goal).
* --teacher_dir / --handoff_dir   v4 teacher (1159062) and two-expert handoff (1153366) on the same protocol.

Main comparison (decided 2026-11-19): cycles of ``up,turn`` per 600 steps; condition = intent x refresh
(hold 1 = 60 Hz, hold 6 = 10 Hz). Welch t-tests on seeds, n = 3 per condition, uncorrected.
``any`` goals are reported separately: per-kind completions per 600 steps in the one-kind conditions.

Checks (status FAIL if any fails): 15 runs = 5 conditions x seeds 0-2, every JSON present; every run evaluated
with the same teacher and seed protocol (1024 envs, 600 steps, seed 20261006, the teacher's sha256);
unified goal reconstruction error <= 2e-3; diagnostic: goal-7 quaternion error exactly 0 and whole-intent error
<= 1e-5 in every step, and its iteration-5000 runs equal the grid's (same phases per 600 steps).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

CONDITIONS = ("random", "any", "up", "up-turn", "any-up-turn")
GROUPS = ("blind", "goal-7", "goal-7@h6", "hand-8", "hand-8@h6")
TESTS = (("hand-8", "goal-7"), ("hand-8@h6", "goal-7@h6"), ("goal-7@h6", "goal-7"), ("hand-8@h6", "hand-8"),
         ("goal-7", "blind"), ("hand-8", "blind"))
CONTROL_HZ = 60.0
V4_TEACHER_SHA256 = "ca3d9367a648a933a927e612697e2aa3fa37e66d564cff76c6ea7c8e250ca1d4"  # frozen v4 seed 0


def per_600(data: dict) -> float:
    """Cycles per 600 steps for a fixed multi-phase cycle, phases per 600 steps for one-kind / random."""
    value = data["cycles_completed_per_600_steps"]
    return value if value is not None and len(data["cycle"]) > 1 else data["phases_completed_per_600_steps"]


def group_name(student: dict) -> str:
    return student["intent"] + ("" if student["intent_hold"] == 1 else f"@h{student['intent_hold']}")


def mean_sd(values: list[float]) -> dict:
    return {"n": len(values), "mean": statistics.mean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else None, "values": values}


def welch(a: list[float], b: list[float]) -> dict:
    from scipy import stats

    result = stats.ttest_ind(a, b, equal_var=False)
    return {"t": float(result.statistic), "p": float(result.pvalue)}


def load_grid(grid_dir: Path, problems: list[str]) -> list[dict]:
    runs = []
    for task in sorted(grid_dir.glob("task*"), key=lambda p: int(p.name[4:])):
        entry = {"task": int(task.name[4:])}
        for condition in CONDITIONS:
            path = task / f"{condition}.json"
            if not path.is_file():
                problems.append(f"missing {path}")
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            student = data["student"]
            entry.setdefault("group", group_name(student))
            entry.setdefault("seed", student["seed"])
            entry.setdefault("teacher_sha256", student["teacher_sha256"])
            if group_name(student) != entry["group"] or student["seed"] != entry["seed"]:
                problems.append(f"{path}: student differs from the other conditions of task {entry['task']}")
            if (data["num_envs"], data["steps"], data["seed"]) != (1024, 600, 20261006):
                problems.append(f"{path}: protocol {data['num_envs']} envs / {data['steps']} steps / seed {data['seed']}")
            if student["unified_goal_check_max_error"] > 2e-3:
                problems.append(f"{path}: unified goal check {student['unified_goal_check_max_error']:.2e}")
            entry[condition] = per_600(data)
            entry[f"{condition}_falls"] = data["falls_per_600_steps"]
            for kind, stats in data["by_kind"].items():
                entry[f"{condition}:{kind}_completed"] = stats["completed_per_600_steps"]
                entry[f"{condition}:{kind}_stalled"] = stats["stalled_fraction_of_started"]
                entry[f"{condition}:{kind}_median_steps"] = stats["steps_to_complete"].get("median")
        runs.append(entry)
    found = sorted((r["group"], r["seed"]) for r in runs if "group" in r)
    expected = sorted((g, s) for g in GROUPS for s in (0, 1, 2))
    if found != expected:
        problems.append(f"runs found {found}, expected {expected}")
    # blind and goal-7 do not load the teacher (no expert in their intent), so only hand-8 runs record it
    teachers = {r.get("teacher_sha256") for r in runs if r.get("group", "").startswith("hand-")}
    if teachers != {V4_TEACHER_SHA256}:
        problems.append(f"hand-8 runs used teachers {teachers}, expected {V4_TEACHER_SHA256}")
    if any(r.get("teacher_sha256") for r in runs if not r.get("group", "").startswith("hand-")):
        problems.append("a blind / goal-7 run records a teacher")
    return runs


def load_diag(diag_dir: Path, grid: list[dict], problems: list[str]) -> list[dict]:
    rows = []
    for path in sorted(diag_dir.glob("task*/*-it*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        student = data["student"]
        check = student.get("goal7_intent_check") or {}
        if not check:
            problems.append(f"{path}: no goal7_intent_check")
        for kind, entry in check.items():
            if entry["max_abs_error_quat"] != 0.0 or entry["max_abs_error"] > 1e-5 or entry["steps_error_over_1e-3"]:
                problems.append(f"{path}: goal-7 intent differs from the env goal in {kind} phases: {entry}")
        rows.append({"seed": student["seed"], "iteration": student["iteration_requested"], "kind": data["cycle"][0],
                     "completed_per_600_steps": data["phases_completed_per_600_steps"],
                     "stalled_fraction": data["by_kind"][data["cycle"][0]]["stalled_fraction_of_started"],
                     "falls_per_600_steps": data["falls_per_600_steps"],
                     "max_intent_error": max((e["max_abs_error"] for e in check.values()), default=None)})
    for row in rows:
        if row["iteration"] != 5000:
            continue
        match = [r for r in grid if r.get("group") == "goal-7" and r.get("seed") == row["seed"]]
        if not match or match[0].get(row["kind"]) != row["completed_per_600_steps"]:
            problems.append(f"diagnostic goal-7 seed {row['seed']} {row['kind']} at 5000 differs from the grid")
    if len(rows) != 30:
        problems.append(f"diagnostic: {len(rows)} runs, expected 30 (3 seeds x 5 iterations x any/up)")
    return rows


def figure(grid: list[dict], diag: list[dict], refs: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ink, muted, grid_color = "#1f1f1e", "#6b6a66", "#e6e5e0"
    colors = {"goal-7": "#2a78d6", "hand-8": "#eb6834"}  # categorical slots 1-2
    kind_colors = {"up": "#1baf7a", "any": "#6250d6"}  # slots 3-4 (different job: goal kind, not intent)
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted,
                         "ytick.color": muted, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": grid_color, "grid.linewidth": 0.8, "axes.axisbelow": True,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1.0, 1.15]})

    # A: up,turn cycles vs intent refresh rate, seeds as dots, means joined.
    xs = {1: 0.0, 6: 1.0}
    for intent, offset in (("goal-7", -0.06), ("hand-8", 0.06)):
        means = []
        for hold in (1, 6):
            group = intent if hold == 1 else f"{intent}@h{hold}"
            values = [r["up-turn"] for r in grid if r["group"] == group]
            x = xs[hold] + offset
            a.scatter([x] * len(values), values, s=36, color=colors[intent], alpha=0.55, lw=0, zorder=3)
            means.append((x, statistics.mean(values)))
        a.plot(*zip(*means), color=colors[intent], lw=2, marker="o", ms=9, mec="white", mew=2, zorder=4)
        a.text(means[-1][0] + 0.07, means[-1][1], intent, color=ink, fontsize=10, va="center")
    blind = [r["up-turn"] for r in grid if r["group"] == "blind"]
    a.scatter([xs[1] - 0.18] * 3, blind, s=36, color=muted, marker="s", lw=0, zorder=3)
    a.text(xs[1] - 0.22, max(blind) + 0.8, "blind", color=muted, fontsize=10, ha="center")
    for name, value in refs.items():
        a.axhline(value, color=muted, ls="--", lw=1.2, zorder=1)
        a.text(1.42, value, f"{name} {value:.1f}", color=muted, fontsize=9, va="bottom", ha="right")
    a.set_xticks([0, 1], [f"every step\n({CONTROL_HZ:.0f} Hz)", f"every 6 steps\n({CONTROL_HZ / 6:.0f} Hz)"])
    a.set_xlim(-0.4, 1.45)
    a.set_ylim(bottom=-0.6)
    a.set_ylabel("up,turn cycles per 600 steps")
    a.set_title("Up + turn cycles by intent refresh rate", loc="left", fontsize=12, color=ink)

    # B: goal-7 (hold 1) during training: any vs up goals completed.
    for kind in ("up", "any"):
        by_seed = {}
        for row in diag:
            if row["kind"] == kind:
                by_seed.setdefault(row["seed"], []).append((row["iteration"], row["completed_per_600_steps"]))
        for points in by_seed.values():
            b.plot(*zip(*sorted(points)), color=kind_colors[kind], lw=1, alpha=0.4, zorder=2)
        iterations = sorted({p[0] for pts in by_seed.values() for p in pts})
        means = [statistics.mean(dict(pts)[it] for pts in by_seed.values()) for it in iterations]
        b.plot(iterations, means, color=kind_colors[kind], lw=2, marker="o", ms=8, mec="white", mew=2, zorder=3)
        label = "up goals (yaw only)" if kind == "up" else "any goals (3-D flip)"
        b.text(iterations[-1] + 120, means[-1], label, color=ink, fontsize=10, va="center")
    b.set_xlim(800, 6600)
    b.set_ylim(bottom=-0.3)
    b.set_xlabel("training iteration")
    b.set_ylabel("goals completed per 600 steps")
    b.set_title("goal-7 student (every step): any goals never learned", loc="left", fontsize=12, color=ink)

    handles = [Line2D([], [], lw=2, marker="o", ms=8, color=colors[k], mec="white", label=k) for k in colors]
    handles += [Line2D([], [], lw=0, marker="s", ms=7, color=muted, label="blind"),
                Line2D([], [], lw=1, color=muted, alpha=0.55, label="single seed (thin / dots)"),
                Line2D([], [], lw=2, color=kind_colors["up"], label="up goals"),
                Line2D([], [], lw=2, color=kind_colors["any"], label="any goals")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid_dir", type=Path, required=True)
    parser.add_argument("--diag_dir", type=Path, required=True)
    parser.add_argument("--teacher_dir", type=Path, required=True)
    parser.add_argument("--handoff_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--no_figure", action="store_true")
    args = parser.parse_args()

    problems: list[str] = []
    grid = load_grid(args.grid_dir, problems)
    diag = load_diag(args.diag_dir, grid, problems)
    teacher = {c: per_600(json.loads((args.teacher_dir / f"{c}.json").read_text())) for c in CONDITIONS}
    handoff = {c: per_600(json.loads((args.handoff_dir / f"{c}.json").read_text())) for c in ("up-turn", "any-up-turn")}

    summary = {}
    for group in GROUPS:
        runs = [r for r in grid if r.get("group") == group]
        summary[group] = {c: mean_sd([r[c] for r in runs]) for c in CONDITIONS}
        for kind in ("any", "up"):
            summary[group][f"{kind}_stalled_fraction"] = mean_sd([r[f"{kind}:{kind}_stalled"] for r in runs])
        summary[group]["up-turn:turn_stalled_fraction"] = mean_sd([r["up-turn:turn_stalled"] for r in runs])
        summary[group]["up-turn_falls"] = mean_sd([r["up-turn_falls"] for r in runs])
        summary[group]["up-turn_retained_vs_teacher"] = summary[group]["up-turn"]["mean"] / teacher["up-turn"]
    tests = {}
    for left, right in TESTS:
        for condition in ("up-turn", "any-up-turn"):
            a = [r[condition] for r in grid if r.get("group") == left]
            b = [r[condition] for r in grid if r.get("group") == right]
            tests[f"{condition}: {left} vs {right}"] = {**welch(a, b), "mean_left": statistics.mean(a),
                                                       "mean_right": statistics.mean(b)}
    retained_h6 = {i: summary[f"{i}@h6"]["up-turn"]["mean"] / summary[i]["up-turn"]["mean"] for i in ("goal-7", "hand-8")}
    status = "PASS" if not problems else "FAIL"
    report = {
        "record_type": "e3_unified_students_analysis",
        "sources": {"grid": "evaluations 1160763_0-14 of training 1160744_0-14", "diagnostic": "1163578_0-2",
                    "teacher": "1159062 (v4 seed 0)", "handoff": "1153366"},
        "main_metric": "up,turn cycles per 600 steps (1024 envs, 600 steps, seed 20261006, last checkpoint)",
        "teacher_v4": teacher, "handoff_two_expert": handoff,
        "by_condition": summary, "welch_uncorrected_n3": tests,
        "up-turn_retained_at_hold6_vs_hold1": retained_h6,
        "goal7_any_diagnostic": diag,
        "problems": problems, "status": status,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "e3_unified_students.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    columns = ["task", "group", "seed", *CONDITIONS, "up-turn_falls", "up-turn:turn_stalled", "any:any_stalled",
               "up:up_stalled", "up-turn:turn_median_steps", "up-turn:up_median_steps"]
    with (args.output_dir / "e3_unified_students_runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for run in grid:
            writer.writerow([run.get(c) for c in columns])
    with (args.output_dir / "e3_goal7_any_diagnostic.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(diag[0]) if diag else ["seed"])
        writer.writeheader()
        writer.writerows(sorted(diag, key=lambda r: (r["kind"], r["seed"], r["iteration"])))
    if not args.no_figure:
        figure(grid, diag, {"v4 teacher": teacher["up-turn"], "two-expert handoff": handoff["up-turn"]},
               args.output_dir / "e3_unified_students.png")

    print(f"{'condition':<11}" + "".join(f"{c:>13}" for c in CONDITIONS))
    for group in GROUPS:
        print(f"{group:<11}" + "".join(f"{summary[group][c]['mean']:>8.2f}±{summary[group][c]['sd']:<4.2f}" for c in CONDITIONS))
    print(f"{'v4 teacher':<11}" + "".join(f"{teacher[c]:>13.2f}" for c in CONDITIONS))
    for name, test in tests.items():
        print(f"welch {name}: {test['mean_left']:.2f} vs {test['mean_right']:.2f} p={test['p']:.3f}")
    print(f"up-turn retained at hold 6: {json.dumps({k: round(v, 3) for k, v in retained_h6.items()})}")
    for problem in problems:
        print(f"problem: {problem}")
    print(f"e3_unified_students_analysis_status={status}")


if __name__ == "__main__":
    main()
