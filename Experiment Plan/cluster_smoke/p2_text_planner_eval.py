#!/usr/bin/env python3
"""Priority 2.2 (Z2 / Plan A baseline): Zero-shot privileged text-state planning with Qwen3-VL-8B-Instruct.

Evaluates whether the base VLM can plan multi-step cube manipulation when given the exact discrete state in text
(isolating planning from visual perception).

Protocol (GOAL_LEVEL_RL_PLAN.md section Z2 & P2.2):
  - Tasks: stratified by optimal BFS length L in {1, 2, 3, 4}, N tasks per level (default 50, total 200 tasks).
  - Decoding modes evaluated in the same job:
      1. `greedy` (temperature 0.0, 1 rollout per task): deterministic accuracy & error taxonomy.
      2. `grpo_group` (temperature 1.0, G=8 rollouts per task): measures mean success rate p and the GRPO
         **effective group fraction** (fraction of tasks with 0 < successes < G, i.e. non-zero GRPO advantage).
  - Episode loop (up to 2*L + 2 rounds):
      - All active rollouts at round r are batched into a single `llm.chat()` call for high throughput.
      - Each round prompt provides:
          * Task rules (24 orientations identified by `(up_colour, front_colour)` of orthogonal faces, 4 detents
            `0, 1, 2, 3` where turning `+90` increases detent by 1 mod 4 and turning `-90` decreases by 1 mod 4;
            CRITICAL CONSTRAINT: `turn` is ONLY legal when `up_colour == "white"`).
          * Target state description and current state description, plus the history of actions taken so far.
      - Model outputs brief reasoning and a JSON action:
          `{"action": "reorient", "up": "<colour>", "front": "<colour>"}` or
          `{"action": "turn", "delta": "+90" | "-90"}`
      - Step attribution per round:
          * `optimal`: legal action that reduces BFS distance to target by 1.
          * `suboptimal_legal`: legal action that does not reduce BFS distance (or keeps/increases it).
          * `illegal_turn_not_white`: attempted `turn` when `up_colour != "white"`.
          * `illegal_orient_pair`: attempted `reorient` to non-orthogonal or invalid `(up, front)` pair.
          * `parse_error`: could not parse a valid JSON action (or truncated).
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import time
from pathlib import Path

from cube_state_graph import (
    COLOUR_PAIR_TO_ORIENT,
    DIST_TABLE,
    DIST_TABLE_LOCAL90,
    ELEMENTARY_ROT_MATS,
    ORIENTATIONS,
    optimal_actions,
    optimal_actions_local90,
    sample_stratified_tasks,
    sample_stratified_tasks_local90,
    shortest_distance,
    state_to_dict,
    state_to_index,
    step_symbolic,
    step_symbolic_local90,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--tasks_per_level", type=int, default=50)
parser.add_argument("--levels", default="1,2,3,4")
parser.add_argument("--group_size", type=int, default=8, help="Rollouts per task for temperature=1.0 GRPO group check")
parser.add_argument("--max_tokens", type=int, default=1024)
parser.add_argument("--max_model_len", type=int, default=16384)
parser.add_argument("--gpu_mem_gib", type=float, default=40.0)
parser.add_argument("--seed", type=int, default=20261120)
parser.add_argument("--plan_variant", default="numeric", choices=["numeric", "relational", "local90"],
                    help="numeric: baseline with explicit detent=0..3 + teleport reorient; "
                         "relational (Lever 1): no detent numbers or lookup table, only colour adjacency strips; "
                         "local90 (Lever 2): reorient restricted to elementary +-90 deg roll/pitch/yaw axes (SO(3))")
args = parser.parse_args()

OPPOSITE_COLOURS = {
    "white": "magenta", "magenta": "white",
    "red": "yellow", "yellow": "red",
    "green": "blue", "blue": "green",
}

SYSTEM_PROMPT = """You are a robot manipulation planner controlling a dexterous hand holding a two-layer hinged cube.

### Cube Structure & State Representation
1. **Six Face Colours (3 opposite pairs)**:
   - `white` (the rotatable top layer) is opposite `magenta` (the bottom face).
   - `red` is opposite `yellow`.
   - `green` is opposite `blue`.
2. **Orientation**: Any of the 24 cube orientations is uniquely specified by two orthogonal faces:
   - `up`: the colour of the face pointing UP (away from the palm).
   - `front`: the colour of the face pointing FRONT (toward camera C1). Since `up` and `front` are orthogonal, `front` can NEVER be the same as `up` or opposite to `up`.
3. **Top-Layer Detent (`detent` in {0, 1, 2, 3})**:
   - The white top layer can rotate in 90-degree clicks relative to the lower 2/3 base:
     - `detent = 0`: aligned (0 deg; red strip is above red base).
     - `detent = 1`: shifted +90 deg (blue strip is above red base).
     - `detent = 2`: shifted 180 deg (yellow strip is above red base).
     - `detent = 3`: shifted -90 deg (green strip is above red base).
   - A `"+90"` turn changes `detent` to `(detent + 1) % 4`.
   - A `"-90"` turn changes `detent` to `(detent - 1) % 4`.

### Legal Macro-Actions (Choose ONE per round)
1. **Reorient the whole cube** (changes `up` and `front` to any valid orthogonal pair in 1 round; keeps `detent` unchanged):
   `{"action": "reorient", "up": "<colour>", "front": "<colour>"}`
2. **Turn the top layer by +90 or -90 degrees** (changes `detent` by +1 or -1 mod 4; keeps `up` and `front` unchanged):
   `{"action": "turn", "delta": "+90"}` or `{"action": "turn", "delta": "-90"}`
   - **CRITICAL PHYSICAL CONSTRAINT**: You can ONLY execute `"turn"` when the white top layer is currently facing UP (`current.up == "white"`). If `current.up != "white"`, you MUST first reorient the cube so that `"up": "white"` (for example `{"action": "reorient", "up": "white", "front": "red"}`), and then turn the top layer in the next round!

Think briefly (2-4 sentences) to compare `current` and `target`, check whether `detent` needs to change, and output your chosen action as a single JSON object at the end of your response."""


SYSTEM_PROMPT_RELATIONAL = """You are a robot manipulation planner controlling a dexterous hand holding a two-layer hinged cube.

### Cube Structure & Spatial Layout
1. **Six Face Colours (3 opposite pairs)**:
   - `white` (the rotatable top layer) is opposite `magenta` (the bottom face).
   - `red` is opposite `yellow`.
   - `green` is opposite `blue`.
   - Looking down at the `white` top face from above (`+z`), the four side faces of the lower 2/3 base in counter-clockwise order are: **`red` -> `green` -> `yellow` -> `blue` -> `red`**.
2. **Orientation**: Any of the 24 cube orientations is uniquely specified by two orthogonal faces:
   - `up`: the colour of the face pointing UP (away from the palm).
   - `front`: the colour of the face pointing FRONT (toward camera C1). Since `up` and `front` are orthogonal, `front` can NEVER be the same as `up` or opposite to `up`.
3. **Top-Layer Strip Alignment (No Numeric IDs)**:
   - The 1/3 `white` top layer also has four coloured side strips (`red`, `green`, `yellow`, `blue`) in the same counter-clockwise order.
   - When the top layer is rotated relative to the base, each base side face has one top-layer coloured strip sitting along its white-layer edge (reported as `strips_on_base: {base_colour: strip_colour}`).
   - A `"+90"` turn rotates the white top layer by +90 degrees counter-clockwise (seen from above the white face), shifting which strip sits on each base face.
   - A `"-90"` turn rotates the white top layer by -90 degrees clockwise (seen from above the white face).

### Legal Macro-Actions (Choose ONE per round)
1. **Reorient the whole cube** (changes `up` and `front` to any valid orthogonal pair in 1 round; keeps `strips_on_base` unchanged):
   `{"action": "reorient", "up": "<colour>", "front": "<colour>"}`
2. **Turn the top layer by +90 or -90 degrees** (rotates the white top layer by +90 or -90 deg; keeps `up` and `front` unchanged):
   `{"action": "turn", "delta": "+90"}` or `{"action": "turn", "delta": "-90"}`
   - **CRITICAL PHYSICAL CONSTRAINT**: You can ONLY execute `"turn"` when the white top layer is currently facing UP (`current.up == "white"`). If `current.up != "white"`, you MUST first reorient the cube so that `"up": "white"`, and then turn the top layer in the next round!

Think briefly (2-4 sentences) to compare `current` and `target`, deduce the required 3D rotation or top-layer turn, and output your chosen action as a single JSON object at the end of your response."""

SYSTEM_PROMPT_LOCAL90 = """You are a robot manipulation planner controlling a dexterous hand holding a two-layer hinged cube.

### Cube Structure & World Axes
1. **World Coordinate Frame**:
   - `+z` points **UP** (away from the palm); `-z` points **DOWN** (into the palm).
   - `+x` points **FRONT** (toward camera C1); `-x` points **BACK**.
   - `+y` points **RIGHT** (so `x, y, z` forms a right-handed frame: `front x right = up`); `-y` points **LEFT**.
2. **Six Face Colours (3 opposite pairs)**:
   - `white` (rotatable top layer) is opposite `magenta` (bottom face).
   - `red` is opposite `yellow`; `green` is opposite `blue`.
   - When `up="white"` and `front="red"`, the face pointing `+y` (RIGHT) is `green`, and `-y` (LEFT) is `blue`.
3. **Top-Layer Detent (`detent` in {0, 1, 2, 3})**:
   - `detent = 0` (aligned, 0 deg), `1` (+90 deg), `2` (180 deg), `3` (-90 deg).
   - Turning `"+90"` changes `detent` to `(detent + 1) % 4`; `"-90"` changes `detent` to `(detent - 1) % 4`.

### Legal Macro-Actions (Elementary 90-Degree Steps ONLY)
Unlike a teleporting controller, the hand can only rotate the cube by **90 degrees at a time** about a world axis:
1. **Elementary 90-degree whole-cube rotation** (keeps `detent` unchanged; rotates the cube in the world frame by the right-hand rule):
   - `{"action": "rotate", "axis": "roll", "delta": "+90" | "-90"}` (about world `+x` FRONT axis: `+90` moves `+y -> +z` and `+z -> -y`)
   - `{"action": "rotate", "axis": "pitch", "delta": "+90" | "-90"}` (about world `+y` RIGHT axis: `+90` moves `+z -> +x` and `+x -> -z`)
   - `{"action": "rotate", "axis": "yaw", "delta": "+90" | "-90"}` (about world `+z` UP axis: `+90` moves `+x -> +y` and `+y -> -x`, keeping `up` face unchanged)
2. **Turn the top layer by +90 or -90 degrees** (changes `detent` by +1 or -1 mod 4; keeps orientation unchanged):
   `{"action": "turn", "delta": "+90" | "-90"}`
   - **CRITICAL PHYSICAL CONSTRAINT**: You can ONLY execute `"turn"` when `current.up == "white"`.

Think briefly (2-4 sentences) to plan the sequence of 90-degree axis rotations and top-layer turns, and output ONE JSON action for the current round at the end of your response."""


def active_system_prompt() -> str:
    if args.plan_variant == "relational":
        return SYSTEM_PROMPT_RELATIONAL
    if args.plan_variant == "local90":
        return SYSTEM_PROMPT_LOCAL90
    return SYSTEM_PROMPT


def format_state_line(s: dict) -> str:
    if args.plan_variant == "relational":
        strips = ", ".join(f"{k}_base->{v}_strip" for k, v in s["strips_by_base"].items())
        return f"`up = \"{s['up_colour']}\"`, `front = \"{s['front_colour']}\"`, `strips_on_base = {{{strips}}}`"
    return (
        f"`up = \"{s['up_colour']}\"`, `front = \"{s['front_colour']}\"`, `detent = {s['detent']}` "
        f"(top band on red side: `{s['top_band_on_red']}`)"
    )


def build_round_prompt(cur: tuple[int, int], dst: tuple[int, int], history: list[dict], round_idx: int, max_rounds: int) -> str:
    c = state_to_dict(*cur)
    t = state_to_dict(*dst)
    hist_lines = []
    for h in history:
        if args.plan_variant == "relational":
            st_str = f"({h['state']['up']}, {h['state']['front']}, red_base->{h['state'].get('red_strip')})"
        else:
            st_str = f"({h['state']['up']}, {h['state']['front']}, detent={h['state']['detent']})"
        hist_lines.append(
            f"  Round {h['round']}: state={st_str} -> action={json.dumps(h['raw_action'])} ({h['status']})"
        )
    hist_str = "\n".join(hist_lines) if hist_lines else "  (None yet — this is Round 1)"
    if args.plan_variant == "local90":
        act_hint = (
            'Output brief reasoning followed by one JSON action '
            '(`{"action": "rotate", "axis": "roll"|"pitch"|"yaw", "delta": "+90"|"-90"}` or '
            '`{"action": "turn", "delta": "+90"|"-90"}`).'
        )
    else:
        act_hint = (
            'Output brief reasoning followed by one JSON action '
            '(`{"action": "reorient", "up": "...", "front": "..."}` or '
            '`{"action": "turn", "delta": "+90" | "-90"}`).'
        )
    return (
        f"### Episode Goal\n"
        f"- Target State: {format_state_line(t)}\n\n"
        f"### History\n{hist_str}\n\n"
        f"### Current State (Round {round_idx} of {max_rounds})\n"
        f"- Current State: {format_state_line(c)}\n\n"
        f"{act_hint}"
    )


def parse_action(text: str, cur: tuple[int, int]) -> tuple[tuple[str, int | str] | None, dict | None, str]:
    """Parse model completion into (symbolic_action, raw_json_dict, status_tag)."""
    cleaned = text.rsplit("</think>", 1)[-1].strip()
    candidates = re.findall(r"\{[^{}]*\}", cleaned)
    if not candidates:
        return None, None, "parse_error"
    obj = None
    for cand in reversed(candidates):
        try:
            parsed = json.loads(cand)
            if isinstance(parsed, dict) and ("action" in parsed or "reorient" in parsed or "turn" in parsed or "rotate" in parsed):
                obj = parsed
                break
        except Exception:
            continue
    if obj is None:
        return None, None, "parse_error"

    act_type = str(obj.get("action", "")).strip().lower()
    if not act_type:
        for k in ("reorient", "rotate", "turn"):
            if k in obj:
                act_type = k
                break

    if args.plan_variant == "local90":
        if act_type == "rotate":
            axis = str(obj.get("axis", "")).strip().lower()
            delta_raw = str(obj.get("delta", "")).strip().lower()
            d_str = "+90" if delta_raw in ("+90", "90", "+1", "1", "cw") else ("-90" if delta_raw in ("-90", "-1", "ccw") else "")
            cmd = f"{axis}:{d_str}"
            if cmd not in ELEMENTARY_ROT_MATS:
                return None, obj, "parse_error"
            return ("rotate", cmd), obj, "ok"
        if act_type == "turn":
            delta_raw = str(obj.get("delta", obj.get("turn", ""))).strip().lower()
            d_str = "+90" if delta_raw in ("+90", "90", "+1", "1", "cw") else ("-90" if delta_raw in ("-90", "-1", "ccw") else "")
            if not d_str:
                return None, obj, "parse_error"
            if not ORIENTATIONS[cur[0]]["is_top_up"]:
                return None, obj, "illegal_turn_not_white"
            return ("turn", d_str), obj, "ok"
        return None, obj, "parse_error"

    if act_type == "reorient":
        up_c = str(obj.get("up", "")).strip().lower()
        front_c = str(obj.get("front", "")).strip().lower()
        if (up_c, front_c) not in COLOUR_PAIR_TO_ORIENT:
            return None, obj, "illegal_orient_pair"
        target_oid = COLOUR_PAIR_TO_ORIENT[(up_c, front_c)]
        return ("reorient", target_oid), obj, "ok"

    if act_type == "turn":
        delta_raw = str(obj.get("delta", obj.get("turn", ""))).strip().lower()
        if delta_raw in ("+90", "90", "+1", "1", "cw", "+90 deg"):
            delta = +1
        elif delta_raw in ("-90", "-1", "ccw", "-90 deg"):
            delta = -1
        else:
            return None, obj, "parse_error"
        if not ORIENTATIONS[cur[0]]["is_top_up"]:
            return None, obj, "illegal_turn_not_white"
        return ("turn", delta), obj, "ok"

    return None, obj, "parse_error"


def run_batch_episodes(llm, tasks: list[dict], num_samples: int, temperature: float, mode_name: str, trace_file) -> dict:
    from vllm import SamplingParams

    params = SamplingParams(temperature=temperature, max_tokens=args.max_tokens)
    # Build rollout states: (task_idx, sample_idx)
    rollouts = []
    for t_idx, t in enumerate(tasks):
        L = t["optimal_steps"]
        for s_idx in range(num_samples):
            rollouts.append({
                "task_id": t["task_id"],
                "task_idx": t_idx,
                "sample_idx": s_idx,
                "L": L,
                "max_rounds": 2 * L + 2,
                "src": (t["src"]["orient_id"], t["src"]["detent"]),
                "cur": (t["src"]["orient_id"], t["src"]["detent"]),
                "dst": (t["dst"]["orient_id"], t["dst"]["detent"]),
                "active": True,
                "succeeded": False,
                "rounds_used": 0,
                "history": [],
                "step_tags": [],
                "total_tokens": 0,
            })

    max_global_rounds = max(r["max_rounds"] for r in rollouts)
    sys_prompt = active_system_prompt()
    t0 = time.time()
    for rnd in range(1, max_global_rounds + 1):
        active_list = [r for r in rollouts if r["active"] and rnd <= r["max_rounds"]]
        if not active_list:
            break
        messages_batch = []
        for r in active_list:
            user_prompt = build_round_prompt(r["cur"], r["dst"], r["history"], rnd, r["max_rounds"])
            messages_batch.append([
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ])
        outputs = llm.chat(messages_batch, params, use_tqdm=False)
        for r, out in zip(active_list, outputs):
            comp = out.outputs[0]
            r["rounds_used"] += 1
            r["total_tokens"] += len(comp.token_ids)
            cur_before = r["cur"]
            c_dict = state_to_dict(*cur_before)
            sym_act, raw_obj, status = parse_action(comp.text, cur_before)
            if comp.finish_reason == "length" and status == "parse_error":
                status = "truncated"

            if status == "ok" and sym_act is not None:
                if args.plan_variant == "local90":
                    opt_set = set(optimal_actions_local90(cur_before, r["dst"]))
                    step_tag = "optimal" if sym_act in opt_set else "suboptimal_legal"
                    no, nd, ok = step_symbolic_local90(*cur_before, sym_act)
                else:
                    opt_set = set(optimal_actions(cur_before, r["dst"]))
                    step_tag = "optimal" if sym_act in opt_set else "suboptimal_legal"
                    no, nd, ok = step_symbolic(*cur_before, sym_act)
                assert ok
                r["cur"] = (no, nd)
            else:
                step_tag = status

            r["step_tags"].append(step_tag)
            r["history"].append({
                "round": rnd,
                "state": {
                    "up": c_dict["up_colour"],
                    "front": c_dict["front_colour"],
                    "detent": c_dict["detent"],
                    "red_strip": c_dict["strips_by_base"]["red"],
                },
                "raw_action": raw_obj if raw_obj is not None else status,
                "status": step_tag,
                "tokens": len(comp.token_ids),
                "text": comp.text,
            })
            if r["cur"] == r["dst"]:
                r["succeeded"] = True
                r["active"] = False
            elif rnd >= r["max_rounds"]:
                r["active"] = False

    elapsed = time.time() - t0
    for r in rollouts:
        trace_file.write(json.dumps({
            "mode": mode_name,
            "task_id": r["task_id"],
            "sample_idx": r["sample_idx"],
            "L": r["L"],
            "succeeded": r["succeeded"],
            "rounds_used": r["rounds_used"],
            "step_tags": r["step_tags"],
            "total_tokens": r["total_tokens"],
            "history": r["history"],
        }) + "\n")
    trace_file.flush()

    # Aggregate statistics by level L and overall
    by_level = {}
    levels = sorted({r["L"] for r in rollouts})
    for L in levels:
        sub = [r for r in rollouts if r["L"] == L]
        by_task = collections.defaultdict(list)
        for r in sub:
            by_task[r["task_id"]].append(r["succeeded"])
        task_succ_rates = [sum(v) / len(v) for v in by_task.values()]
        effective_groups = sum(0 < sum(v) < len(v) for v in by_task.values())
        any_solved = sum(sum(v) > 0 for v in by_task.values())

        tag_counts = collections.Counter(tag for r in sub for tag in r["step_tags"])
        total_steps = max(sum(tag_counts.values()), 1)
        succ_runs = [r for r in sub if r["succeeded"]]

        # First-step attribution
        first_tags = collections.Counter(r["step_tags"][0] for r in sub if r["step_tags"])
        n_sub = max(len(sub), 1)

        by_level[str(L)] = {
            "n_tasks": len(by_task),
            "n_rollouts": len(sub),
            "success_rate": float(sum(r["succeeded"] for r in sub) / n_sub),
            "pass_at_k": float(any_solved / max(len(by_task), 1)),
            "effective_group_fraction": float(effective_groups / max(len(by_task), 1)) if num_samples > 1 else None,
            "mean_rounds_succeeded": float(sum(r["rounds_used"] for r in succ_runs) / len(succ_runs)) if succ_runs else None,
            "rounds_ratio_vs_optimal": float(sum(r["rounds_used"] for r in succ_runs) / (len(succ_runs) * L)) if succ_runs else None,
            "mean_tokens_per_round": float(sum(r["total_tokens"] for r in sub) / total_steps),
            "step_attribution": {k: round(v / total_steps, 4) for k, v in sorted(tag_counts.items())},
            "first_step_attribution": {k: round(v / n_sub, 4) for k, v in sorted(first_tags.items())},
        }
        print(
            f"p2_text_plan mode={mode_name} L={L} succ={by_level[str(L)]['success_rate']:.3f} "
            f"pass@{num_samples}={by_level[str(L)]['pass_at_k']:.3f} "
            f"eff_group={by_level[str(L)]['effective_group_fraction']} "
            f"steps={by_level[str(L)]['step_attribution']}",
            flush=True,
        )

    all_tags = collections.Counter(tag for r in rollouts for tag in r["step_tags"])
    tot_st = max(sum(all_tags.values()), 1)
    by_task_all = collections.defaultdict(list)
    for r in rollouts:
        by_task_all[r["task_id"]].append(r["succeeded"])
    eff_all = sum(0 < sum(v) < len(v) for v in by_task_all.values()) / max(len(by_task_all), 1)
    return {
        "mode": mode_name,
        "temperature": temperature,
        "num_samples": num_samples,
        "elapsed_s": elapsed,
        "overall_success_rate": float(sum(r["succeeded"] for r in rollouts) / len(rollouts)),
        "overall_effective_group_fraction": float(eff_all) if num_samples > 1 else None,
        "overall_step_attribution": {k: round(v / tot_st, 4) for k, v in sorted(all_tags.items())},
        "by_level": by_level,
    }


def main() -> None:
    import torch
    from vllm import LLM

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    if args.plan_variant == "local90":
        tasks = sample_stratified_tasks_local90(n_per_level=args.tasks_per_level, levels=levels, seed=args.seed)
    else:
        tasks = sample_stratified_tasks(n_per_level=args.tasks_per_level, levels=levels, seed=args.seed)
    (out / "tasks.json").write_text(json.dumps(tasks, indent=2) + "\n", encoding="utf-8")

    free, total = torch.cuda.mem_get_info()
    utilization = 0.9 if args.gpu_mem_gib <= 0 else min(0.92, args.gpu_mem_gib * 2**30 / total)
    print(f"gpu_free_gib={free / 2**30:.1f} vllm_utilization={utilization:.3f} n_tasks={len(tasks)} variant={args.plan_variant}", flush=True)

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=utilization,
        seed=3407,
    )

    traces = open(out / "traces.jsonl", "w", encoding="utf-8")
    greedy_res = run_batch_episodes(llm, tasks, num_samples=1, temperature=0.0, mode_name="greedy", trace_file=traces)
    group_res = run_batch_episodes(
        llm, tasks, num_samples=args.group_size, temperature=1.0, mode_name="grpo_group", trace_file=traces
    )
    traces.close()

    summary = {
        "record_type": "p2_text_planner_eval",
        "plan_variant": args.plan_variant,
        "model": args.model,
        "seed": args.seed,
        "tasks_per_level": args.tasks_per_level,
        "levels": levels,
        "group_size": args.group_size,
        "max_tokens": args.max_tokens,
        "greedy": greedy_res,
        "grpo_group": group_res,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("p2_text_planner_eval_status=PASS", flush=True)


if __name__ == "__main__":
    main()
