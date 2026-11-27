#!/usr/bin/env python3
"""Open-loop 96-state perception and BFS macro-action planning evaluation for C1 SFT models.

Evaluates a merged HF VLM checkpoint via vLLM on the held-out 180-frame `test` split of `z1_frames_v2.jsonl`:
  1. Full 96-state perception (`map_state` for vision_full, `cinv_map_state` for color_invariant) on both
     canonical (`none`) and OOD recolored (`reverse_cycle`: green<->blue swapped) 3-view montages.
  2. Open-loop BFS planning (`plan_L1..L4`, 2 deterministic targets per frame = 360 queries) on canonical
     montages, measuring:
       - `json_valid`: valid JSON with expected fields
       - `state_exact`: exact match on (sky_face, front_face, on_red_panel_strip)
       - `action_exact`: exact match with bfs_plan(cur, dst)[0]
       - `action_optimal`: legal action from TRUE cur state that strictly reduces shortest-path distance
         DIST_TABLE[next, dst] == DIST_TABLE[cur, dst] - 1 (handles multiple optimal paths, e.g. +90/-90 for 180 deg)
       - `plan_given_pred_state_optimal`: whether predicted action is optimal for the model's OWN predicted state
         (separates perception errors from planning/reasoning errors).
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import random
import re
import time
from pathlib import Path

import numpy as np

from build_c1_sft_dataset import (
    SYS_COLOR_INV,
    SYS_VISION_FULL,
    format_action_json,
    make_montage,
    on_red_strip,
)
from cube_state_graph import (
    COLOUR_PAIR_TO_ORIENT,
    DIST_TABLE,
    FACE_COLOUR,
    ORIENTATIONS,
    SIDES,
    bfs_plan,
    index_to_state,
    nearest_orientation,
    state_to_index,
)
from mcq_common import image_url

DETENT_TO_DEG = {0: "0", 1: "+90", 2: "180", 3: "-90"}
STRIP_TO_DETENT = {FACE_COLOUR[SIDES[(-d) % 4]]: d for d in range(4)}
REV_CYCLE_MAP = dict(FACE_COLOUR)
REV_CYCLE_MAP["py"] = "blue"
REV_CYCLE_MAP["ny"] = "green"
NOVEL_PALETTE_MAP = {"px": "orange", "py": "cyan", "nx": "purple", "ny": "pink", "pz": "white", "nz": "magenta"}
NOVEL_TO_CANON = {"orange": "red", "cyan": "green", "purple": "yellow", "pink": "blue"}
CANON_TO_NOVEL = {v: k for k, v in NOVEL_TO_CANON.items()}


def _canon_colour(c: str) -> str:
    c = str(c).strip().lower()
    return NOVEL_TO_CANON.get(c, c)


def encode_pil(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return image_url(buf.getvalue())


def extract_json(text: str) -> dict | None:
    text = text.strip()
    # Strip markdown code fences if present
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        i = text.find("{")
        j = text.rfind("}")
        if i != -1 and j != -1 and j > i:
            text = text[i : j + 1]
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def parse_action(act_obj: dict | None) -> tuple[str, int] | None:
    if not isinstance(act_obj, dict):
        return None
    kind = act_obj.get("action")
    if kind == "turn":
        d = str(act_obj.get("delta", "")).strip()
        if d in ("+90", "90", "+1", "1"):
            return ("turn", 1)
        if d in ("-90", "-1"):
            return ("turn", -1)
        return None
    if kind == "reorient":
        sky = _canon_colour(act_obj.get("sky_face", ""))
        front = _canon_colour(act_obj.get("front_face", ""))
        oid = COLOUR_PAIR_TO_ORIENT.get((sky, front))
        if oid is None:
            return None
        return ("reorient", oid)
    return None


def apply_action(state: tuple[int, int], act: tuple[str, int]) -> tuple[int, int] | None:
    oid, det = state
    kind, val = act
    if kind == "reorient":
        if val == oid or not (0 <= val < 24):
            return None
        return (val, det)
    if kind == "turn":
        if not ORIENTATIONS[oid]["is_top_up"]:
            return None
        if val not in (-1, 1):
            return None
        return (oid, (det + val) % 4)
    return None


def is_action_optimal(cur: tuple[int, int], dst: tuple[int, int], act: tuple[str, int] | None) -> bool:
    if act is None:
        return False
    nxt = apply_action(cur, act)
    if nxt is None:
        return False
    d_cur = int(DIST_TABLE[state_to_index(*cur), state_to_index(*dst)])
    d_nxt = int(DIST_TABLE[state_to_index(*nxt), state_to_index(*dst)])
    return d_cur > 0 and d_nxt == d_cur - 1


def parse_pred_state(state_obj: dict | None) -> tuple[int, int] | None:
    if not isinstance(state_obj, dict):
        return None
    sky = _canon_colour(state_obj.get("sky_face", ""))
    front = _canon_colour(state_obj.get("front_face", ""))
    strip = _canon_colour(state_obj.get("on_red_panel_strip", state_obj.get("on_orange_panel_strip", "")))
    oid = COLOUR_PAIR_TO_ORIENT.get((sky, front))
    det = STRIP_TO_DETENT.get(strip)
    if oid is None or det is None:
        return None
    return (oid, det)


def load_test_frames_with_quat(frames_path: Path) -> list[dict]:
    test_frames = [
        json.loads(line)
        for line in open(frames_path, encoding="utf-8")
        if json.loads(line).get("split") == "test"
    ]
    # Index source frames.jsonl by (run_dir, step, env)
    src_cache: dict[str, dict[tuple[int, int], dict]] = {}
    enriched = []
    for f in test_frames:
        rdir = f["run_dir"]
        if rdir not in src_cache:
            tbl = {}
            with open(Path(rdir) / "frames.jsonl", encoding="utf-8") as sf:
                for line in sf:
                    sr = json.loads(line)
                    tbl[(sr["step"], sr["env"])] = sr
            src_cache[rdir] = tbl
        sr = src_cache[rdir][(f["step"], f["env"])]
        oid, o_err = nearest_orientation(np.asarray(sr["cube_quat_wxyz"], dtype=float))
        det = int(f["detent"]) % 4
        enriched.append({
            "id": f["id"],
            "run_dir": Path(rdir),
            "step": int(f["step"]),
            "env": int(f["env"]),
            "oid": oid,
            "o_err": float(o_err),
            "detent": det,
            "sidx": state_to_index(oid, det),
            "up_tilt_deg": float(f["up_tilt_deg"]),
        })
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--frames", required=True, help="z1_frames_v2.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", required=True, choices=["vision_full", "color_invariant"])
    parser.add_argument("--plans_per_frame", type=int, default=2)
    parser.add_argument("--eval_novel_plan", action="store_true",
                        help="Also evaluate zero-shot open-loop BFS planning on held-out novel_palette (orange/cyan/purple/pink)")
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--gpu_mem_gib", type=float, default=40.0)
    args = parser.parse_args()

    import torch
    from vllm import LLM, SamplingParams

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    frames = load_test_frames_with_quat(Path(args.frames))
    print(f"loaded_test_frames={len(frames)} clean_oerr_le28={sum(1 for f in frames if f['o_err'] <= 28.0)}", flush=True)

    # Pre-render canonical & OOD reverse_cycle montages
    urls_canon = {}
    urls_ood = {}
    for f in frames:
        urls_canon[f["id"]] = encode_pil(make_montage(f["run_dir"], f["step"], f["env"]))
        urls_ood[f["id"]] = encode_pil(make_montage(f["run_dir"], f["step"], f["env"], side_colour_map=REV_CYCLE_MAP))

    free, total = torch.cuda.mem_get_info()
    utilization = 0.9 if args.gpu_mem_gib <= 0 else args.gpu_mem_gib * 2**30 / total
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=utilization,
        seed=3407,
    )
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    summary = {"model": args.model, "mode": args.mode, "n_frames": len(frames), "tasks": {}}

    # --- Task 1: Native 96-state perception (canonical + OOD reverse_cycle) ---
    for tag, urls, cmap in [("state_canonical", urls_canon, FACE_COLOUR), ("state_ood_rev_cycle", urls_ood, REV_CYCLE_MAP)]:
        t0 = time.time()
        msgs_list = []
        for f in frames:
            if args.mode == "vision_full":
                sys_prompt = SYS_VISION_FULL
                user_text = (
                    "Inspect the 3-camera montage and report `view_obs` (for each camera, list visible coloured side "
                    "panels as `large_2_3` and `small_1_3`) and the global `state` (`sky_face`, `front_face`, "
                    "`on_red_panel_strip`) in JSON."
                )
            else:
                sys_prompt = SYS_COLOR_INV
                user_text = (
                    "Inspect the 3-camera montage. Group each visible side face by 3D coplanarity (`large_2_3` base "
                    "and `small_1_3` cap strip in the same plane), deduce the colour-agnostic relative cap rotation "
                    "`cap_relative_turn_deg` (\"0\", \"+90\", \"180\", \"-90\"), and report `state` in JSON."
                )
            msgs_list.append([
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": urls[f["id"]]}},
                    {"type": "text", "text": user_text},
                ]},
            ])
        outputs = llm.chat(msgs_list, params, use_tqdm=False)
        stats = collections.Counter()
        rows_out = []
        for f, res in zip(frames, outputs):
            txt = res.outputs[0].text
            obj = extract_json(txt)
            st = obj.get("state") if isinstance(obj, dict) else None
            o_info = ORIENTATIONS[f["oid"]]
            det = f["detent"]
            t_sky = cmap[o_info["up_face"]]
            t_front = cmap[o_info["front_face"]]
            t_strip = cmap[SIDES[(-det) % 4]]
            t_deg = DETENT_TO_DEG[det]

            valid = isinstance(st, dict)
            p_sky = str(st.get("sky_face", "")).strip().lower() if valid else ""
            p_front = str(st.get("front_face", "")).strip().lower() if valid else ""
            if args.mode == "vision_full":
                p_strip = str(st.get("on_red_panel_strip", "")).strip().lower() if valid else ""
                ok_cap = p_strip == t_strip
            else:
                p_deg = str(st.get("cap_relative_turn_deg", "")).strip() if valid else ""
                p_strip = str(st.get("on_anchor_panel_strip", "")).strip().lower() if valid else ""
                ok_cap = (p_deg == t_deg)
                stats["anchor_strip_ok"] += int(p_strip == t_strip)

            ok_sky = p_sky == t_sky
            ok_front = p_front == t_front
            ok_all = ok_sky and ok_front and ok_cap

            stats["n"] += 1
            stats["json_valid"] += int(valid)
            stats["sky_ok"] += int(ok_sky)
            stats["front_ok"] += int(ok_front)
            stats["cap_ok"] += int(ok_cap)
            stats["state_exact"] += int(ok_all)
            rows_out.append({
                "id": f["id"], "o_err": f["o_err"], "detent": det,
                "truth": {"sky_face": t_sky, "front_face": t_front, "strip": t_strip, "deg": t_deg},
                "pred_state": st, "ok_sky": ok_sky, "ok_front": ok_front, "ok_cap": ok_cap, "ok_all": ok_all,
                "text": txt,
            })
        n = max(stats["n"], 1)
        rec = {
            "n": stats["n"],
            "seconds": round(time.time() - t0, 1),
            "json_valid": round(stats["json_valid"] / n, 4),
            "sky_acc": round(stats["sky_ok"] / n, 4),
            "front_acc": round(stats["front_ok"] / n, 4),
            "cap_acc": round(stats["cap_ok"] / n, 4),
            "state_exact_96": round(stats["state_exact"] / n, 4),
        }
        if args.mode == "color_invariant":
            rec["anchor_strip_acc"] = round(stats["anchor_strip_ok"] / n, 4)
        summary["tasks"][tag] = rec
        (out / f"{tag}_responses.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows_out), encoding="utf-8"
        )
        print(f"openloop_{tag}=" + json.dumps(rec), flush=True)

    # --- Task 2: Open-loop BFS Planning (L=1..4, canonical montage) ---
    rng = random.Random(args.seed)
    plan_queries = []
    for f in frames:
        sidx = f["sidx"]
        for p_i in range(args.plans_per_frame):
            L = ((len(plan_queries)) % 4) + 1  # balanced L=1,2,3,4
            cands = np.nonzero(DIST_TABLE[sidx] == L)[0]
            if len(cands) == 0:
                cands = np.nonzero(DIST_TABLE[sidx] > 0)[0]
            dst_idx = int(rng.choice(cands.tolist()))
            dst_o, dst_d = index_to_state(dst_idx)
            plan = bfs_plan((f["oid"], f["detent"]), (dst_o, dst_d))
            dst_state = {
                "sky_face": ORIENTATIONS[dst_o]["up_colour"],
                "front_face": ORIENTATIONS[dst_o]["front_colour"],
                "on_red_panel_strip": on_red_strip(dst_d),
            }
            plan_queries.append({
                "frame": f, "L": int(DIST_TABLE[sidx, dst_idx]),
                "dst": (dst_o, dst_d), "dst_state": dst_state,
                "bfs_act": plan[0], "bfs_act_json": format_action_json(plan[0]),
            })

    t0 = time.time()
    plan_msgs = []
    for q in plan_queries:
        f = q["frame"]
        plan_msgs.append([
            {"role": "system", "content": SYS_VISION_FULL},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": urls_canon[f["id"]]}},
                {"type": "text", "text": (
                    f"Target State: {json.dumps(q['dst_state'])}\n"
                    "Observe the 3-camera montage, output `view_obs`, `state`, `thought`, and the next optimal `action` in JSON."
                )},
            ]},
        ])
    outputs = llm.chat(plan_msgs, params, use_tqdm=False)

    p_stats = collections.Counter()
    by_L = collections.defaultdict(collections.Counter)
    by_kind = collections.defaultdict(collections.Counter)
    plan_rows = []
    for q, res in zip(plan_queries, outputs):
        f = q["frame"]
        cur = (f["oid"], f["detent"])
        dst = q["dst"]
        txt = res.outputs[0].text
        obj = extract_json(txt)
        valid_json = isinstance(obj, dict)
        pred_state = parse_pred_state(obj.get("state") if valid_json else None)
        pred_act = parse_action(obj.get("action") if valid_json else None)

        state_ok = pred_state == cur
        act_exact = pred_act == q["bfs_act"]
        act_opt = is_action_optimal(cur, dst, pred_act)
        act_opt_given_pred = (pred_state is not None) and is_action_optimal(pred_state, dst, pred_act)

        L = q["L"]
        kind = q["bfs_act"][0]
        for bucket in (p_stats, by_L[L], by_kind[kind]):
            bucket["n"] += 1
            bucket["json_valid"] += int(valid_json)
            bucket["act_parsed"] += int(pred_act is not None)
            bucket["state_exact"] += int(state_ok)
            bucket["action_exact"] += int(act_exact)
            bucket["action_optimal"] += int(act_opt)
            bucket["plan_given_pred_valid"] += int(pred_state is not None)
            bucket["plan_given_pred_optimal"] += int(act_opt_given_pred)

        plan_rows.append({
            "id": f["id"], "L": L, "cur": cur, "dst": dst,
            "bfs_act": q["bfs_act_json"],
            "pred_state": obj.get("state") if valid_json else None,
            "pred_action": obj.get("action") if valid_json else None,
            "state_ok": state_ok, "act_exact": act_exact, "act_optimal": act_opt,
            "act_opt_given_pred": act_opt_given_pred, "text": txt,
        })

    def summarize(c: collections.Counter) -> dict:
        n = max(c["n"], 1)
        npv = max(c["plan_given_pred_valid"], 1)
        return {
            "n": c["n"],
            "json_valid": round(c["json_valid"] / n, 4),
            "act_parsed": round(c["act_parsed"] / n, 4),
            "state_exact_96": round(c["state_exact"] / n, 4),
            "action_exact": round(c["action_exact"] / n, 4),
            "action_optimal": round(c["action_optimal"] / n, 4),
            "plan_given_pred_optimal": round(c["plan_given_pred_optimal"] / npv, 4),
        }

    plan_rec = summarize(p_stats)
    plan_rec["seconds"] = round(time.time() - t0, 1)
    plan_rec["by_L"] = {f"L{k}": summarize(v) for k, v in sorted(by_L.items())}
    plan_rec["by_kind"] = {k: summarize(v) for k, v in sorted(by_kind.items())}
    summary["tasks"]["openloop_plan_canonical"] = plan_rec

    (out / "openloop_plan_responses.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in plan_rows), encoding="utf-8"
    )
    print("openloop_plan_canonical=" + json.dumps(plan_rec), flush=True)

    # --- Task 3 (optional): Zero-shot Open-loop BFS Planning on held-out novel_palette (orange/cyan/purple/pink) ---
    if args.eval_novel_plan:
        urls_novel = {
            f["id"]: encode_pil(make_montage(f["run_dir"], f["step"], f["env"], side_colour_map=NOVEL_PALETTE_MAP))
            for f in frames
        }
        t0_n = time.time()
        novel_msgs = []
        for q in plan_queries:
            f = q["frame"]
            novel_dst = {
                "sky_face": CANON_TO_NOVEL.get(q["dst_state"]["sky_face"], q["dst_state"]["sky_face"]),
                "front_face": CANON_TO_NOVEL.get(q["dst_state"]["front_face"], q["dst_state"]["front_face"]),
                "on_orange_panel_strip": CANON_TO_NOVEL.get(
                    q["dst_state"]["on_red_panel_strip"], q["dst_state"]["on_red_panel_strip"]
                ),
            }
            novel_msgs.append([
                {"role": "system", "content": SYS_VISION_FULL},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": urls_novel[f["id"]]}},
                    {"type": "text", "text": (
                        "This cube uses a novel side-colour palette: orange, cyan, purple, pink "
                        "(top is white, bottom is magenta).\n"
                        f"Target State: {json.dumps(novel_dst)}\n"
                        "Observe the 3-camera montage, output `view_obs`, `state` (with `sky_face`, `front_face`, "
                        "`on_orange_panel_strip`), `thought`, and the next optimal `action` in JSON."
                    )},
                ]},
            ])
        novel_outputs = llm.chat(novel_msgs, params, use_tqdm=False)
        np_stats = collections.Counter()
        n_by_L = collections.defaultdict(collections.Counter)
        n_by_kind = collections.defaultdict(collections.Counter)
        novel_rows = []
        for q, res in zip(plan_queries, novel_outputs):
            f = q["frame"]
            cur = (f["oid"], f["detent"])
            dst = q["dst"]
            txt = res.outputs[0].text
            obj = extract_json(txt)
            valid_json = isinstance(obj, dict)
            pred_state = parse_pred_state(obj.get("state") if valid_json else None)
            pred_act = parse_action(obj.get("action") if valid_json else None)
            state_ok = pred_state == cur
            act_exact = pred_act == q["bfs_act"]
            act_opt = is_action_optimal(cur, dst, pred_act)
            act_opt_given_pred = (pred_state is not None) and is_action_optimal(pred_state, dst, pred_act)
            L = q["L"]
            kind = q["bfs_act"][0]
            for bucket in (np_stats, n_by_L[L], n_by_kind[kind]):
                bucket["n"] += 1
                bucket["json_valid"] += int(valid_json)
                bucket["act_parsed"] += int(pred_act is not None)
                bucket["state_exact"] += int(state_ok)
                bucket["action_exact"] += int(act_exact)
                bucket["action_optimal"] += int(act_opt)
                bucket["plan_given_pred_valid"] += int(pred_state is not None)
                bucket["plan_given_pred_optimal"] += int(act_opt_given_pred)
            novel_rows.append({
                "id": f["id"], "L": L, "cur": cur, "dst": dst,
                "bfs_act": q["bfs_act_json"],
                "pred_state": obj.get("state") if valid_json else None,
                "pred_action": obj.get("action") if valid_json else None,
                "state_ok": state_ok, "act_exact": act_exact, "act_optimal": act_opt,
                "act_opt_given_pred": act_opt_given_pred, "text": txt,
            })
        novel_rec = summarize(np_stats)
        novel_rec["seconds"] = round(time.time() - t0_n, 1)
        novel_rec["by_L"] = {f"L{k}": summarize(v) for k, v in sorted(n_by_L.items())}
        novel_rec["by_kind"] = {k: summarize(v) for k, v in sorted(n_by_kind.items())}
        summary["tasks"]["openloop_plan_novel_palette"] = novel_rec
        (out / "openloop_plan_novel_responses.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in novel_rows), encoding="utf-8"
        )
        print("openloop_plan_novel_palette=" + json.dumps(novel_rec), flush=True)

    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("eval_c1_openloop_plan_status=PASS")


if __name__ == "__main__":
    main()
