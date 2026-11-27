#!/usr/bin/env python3
"""Multi-turn closed-loop evaluation on the 200 frozen test tasks using real 3-view montage images.

Evaluates a merged HF VLM checkpoint via vLLM on the frozen 200 stratified tasks (L=1..4, 50 each) from
`tasks.json`. At each round, renders/samples a real 3-view `montage` image of the current discrete state
(preferring the held-out 180-frame test split pool, falling back to the train image pool only for states
not present in the 180 test frames), queries the VLM with the exact `SYS_VISION_FULL` prompt, applies
the predicted macro-action on the 96-state graph (`cube_state_graph`), and repeats up to `2*L + 2` rounds.
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import random
import time
from pathlib import Path

from build_c1_sft_dataset import SYS_VISION_FULL, make_montage, on_red_strip
from cube_state_graph import DIST_TABLE, ORIENTATIONS, index_to_state, state_to_index
from eval_c1_openloop_plan import (
    apply_action,
    extract_json,
    is_action_optimal,
    load_test_frames_with_quat,
    parse_action,
    parse_pred_state,
)
from mcq_common import image_url


def encode_pil(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return image_url(buf.getvalue())


def encode_file(path: str) -> str:
    return image_url(Path(path).read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tasks", required=True, help="Frozen 200 test tasks.json (from 1169616)")
    parser.add_argument("--test_frames", required=True, help="z1_frames_v2.jsonl (test split images)")
    parser.add_argument("--train_pool", required=True, help="state_image_pool.json fallback for uncovered test states")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--gpu_mem_gib", type=float, default=40.0)
    parser.add_argument("--seed", type=int, default=20261121)
    args = parser.parse_args()

    import torch
    from vllm import LLM, SamplingParams

    rng = random.Random(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tasks = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    train_pool: dict[int, list[str]] = {
        int(k): v for k, v in json.loads(Path(args.train_pool).read_text(encoding="utf-8")).items() if v
    }

    # Build held-out test image pool (data URLs) from the 180 test frames
    test_frames = load_test_frames_with_quat(Path(args.test_frames))
    test_pool: dict[int, list[str]] = collections.defaultdict(list)
    for f in test_frames:
        test_pool[f["sidx"]].append(encode_pil(make_montage(f["run_dir"], f["step"], f["env"])))

    train_url_cache: dict[str, str] = {}

    def get_image_url(sidx: int) -> tuple[str | None, str]:
        if sidx in test_pool:
            return rng.choice(test_pool[sidx]), "test"
        if sidx in train_pool:
            p = rng.choice(train_pool[sidx])
            if p not in train_url_cache:
                train_url_cache[p] = encode_file(p)
            return train_url_cache[p], "train_fallback"
        return None, "uncovered"

    covered_states = set(test_pool.keys()) | set(train_pool.keys())
    print(
        f"pool_coverage test_states={len(test_pool)} train_states={len(train_pool)} "
        f"union_states={len(covered_states)}/96 tasks={len(tasks)}",
        flush=True,
    )

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

    # Initialize episode states
    episodes = []
    for t in tasks:
        src = (int(t["src"]["orient_id"]), int(t["src"]["detent"]))
        dst = (int(t["dst"]["orient_id"]), int(t["dst"]["detent"]))
        L = int(t["optimal_steps"])
        dst_state = {
            "sky_face": ORIENTATIONS[dst[0]]["up_colour"],
            "front_face": ORIENTATIONS[dst[0]]["front_colour"],
            "on_red_panel_strip": on_red_strip(dst[1]),
        }
        episodes.append({
            "task_id": t["task_id"],
            "L": L,
            "max_rounds": 2 * L + 2,
            "src": src,
            "dst": dst,
            "dst_state": dst_state,
            "cur": src,
            "done": False,
            "success": False,
            "uncovered_abort": False,
            "rounds": 0,
            "history": [],
        })

    t0 = time.time()
    max_possible_rounds = max(ep["max_rounds"] for ep in episodes)
    step_counts = collections.Counter()

    for rnd in range(1, max_possible_rounds + 1):
        active = []
        msgs_list = []
        img_sources = []
        for ep in episodes:
            if ep["done"]:
                continue
            if ep["rounds"] >= ep["max_rounds"]:
                ep["done"] = True
                continue
            sidx = state_to_index(*ep["cur"])
            u, src_kind = get_image_url(sidx)
            if u is None:
                ep["done"] = True
                ep["uncovered_abort"] = True
                continue
            active.append(ep)
            img_sources.append(src_kind)
            msgs_list.append([
                {"role": "system", "content": SYS_VISION_FULL},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": u}},
                    {"type": "text", "text": (
                        f"Target State: {json.dumps(ep['dst_state'])}\n"
                        "Observe the 3-camera montage, output `view_obs`, `state`, `thought`, and the next optimal `action` in JSON."
                    )},
                ]},
            ])
        if not active:
            break

        outputs = llm.chat(msgs_list, params, use_tqdm=False)
        for ep, src_kind, res in zip(active, img_sources, outputs):
            ep["rounds"] += 1
            step_counts["total_steps"] += 1
            step_counts[f"img_{src_kind}"] += 1

            txt = res.outputs[0].text
            obj = extract_json(txt)
            valid_json = isinstance(obj, dict)
            pred_state = parse_pred_state(obj.get("state") if valid_json else None)
            pred_act = parse_action(obj.get("action") if valid_json else None)

            state_ok = pred_state == ep["cur"]
            act_opt = is_action_optimal(ep["cur"], ep["dst"], pred_act)
            nxt = apply_action(ep["cur"], pred_act) if pred_act is not None else None
            illegal = (pred_act is not None) and (nxt is None)

            step_counts["json_valid"] += int(valid_json)
            step_counts["state_ok"] += int(state_ok)
            step_counts["act_opt"] += int(act_opt)
            step_counts["illegal_action"] += int(illegal)
            step_counts["parse_fail"] += int(pred_act is None)

            ep["history"].append({
                "round": rnd, "cur": ep["cur"], "img_source": src_kind,
                "pred_state": pred_state, "pred_act": pred_act,
                "state_ok": state_ok, "act_opt": act_opt, "illegal": illegal,
            })
            if nxt is not None:
                ep["cur"] = nxt
            if ep["cur"] == ep["dst"]:
                ep["done"] = True
                ep["success"] = True
            elif ep["rounds"] >= ep["max_rounds"]:
                ep["done"] = True

    def agg(eps: list[dict]) -> dict:
        n = len(eps)
        if n == 0:
            return {"n": 0}
        succ = [e for e in eps if e["success"]]
        uncov = [e for e in eps if e["uncovered_abort"]]
        cov = [e for e in eps if not e["uncovered_abort"]]
        ratios = [e["rounds"] / e["L"] for e in succ]
        return {
            "n": n,
            "n_covered": len(cov),
            "n_uncovered_abort": len(uncov),
            "success_all": round(len(succ) / n, 4),
            "success_covered": round(len(succ) / max(len(cov), 1), 4),
            "mean_rounds_ratio": round(sum(ratios) / len(ratios), 4) if ratios else None,
        }

    ns = max(step_counts["total_steps"], 1)
    summary = {
        "model": args.model,
        "seconds": round(time.time() - t0, 1),
        "overall": agg(episodes),
        "by_L": {f"L{L}": agg([e for e in episodes if e["L"] == L]) for L in (1, 2, 3, 4)},
        "step_metrics": {
            "total_steps": step_counts["total_steps"],
            "img_test_frac": round(step_counts["img_test"] / ns, 4),
            "img_train_fallback_frac": round(step_counts["img_train_fallback"] / ns, 4),
            "json_valid": round(step_counts["json_valid"] / ns, 4),
            "state_exact_96": round(step_counts["state_ok"] / ns, 4),
            "action_optimal": round(step_counts["act_opt"] / ns, 4),
            "illegal_action": round(step_counts["illegal_action"] / ns, 4),
            "parse_fail": round(step_counts["parse_fail"] / ns, 4),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "episodes.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in episodes), encoding="utf-8"
    )
    print("closedloop_pool_summary=" + json.dumps(summary), flush=True)
    print("eval_c2_closedloop_pool_status=PASS")


if __name__ == "__main__":
    main()
