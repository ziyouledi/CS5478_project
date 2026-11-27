#!/usr/bin/env python3
"""Z1: zero-shot cube-state perception of an open VLM on z1_capture.py frames (GOAL_LEVEL_RL_PLAN.md Z1).

For every question (z1_build_questions.py) and camera condition, the model sees the frame from those cameras
and answers {"up": colour, "above_red": colour}. --samples completions per prompt at --temperature (the GRPO
group). Writes responses.jsonl (every completion, parsed and scored) and summary.json: per condition accuracy
of up / above_red / both, parse failures, the informative-group rate (questions with 0 < correct < samples, for
"both"), and accuracy by stratum (up colour, detent, visibility of the least visible camera used).
"""

from __future__ import annotations

import argparse
import base64
import collections
import json
import re
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", required=True, help="Local model directory")
parser.add_argument("--questions", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--conditions", default="c1,s040,s220,top,c1+s040,s040+s220,c1+s040+s220,c1@nohand")
parser.add_argument("--samples", type=int, default=8)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--max_tokens", type=int, default=2048)
parser.add_argument("--max_model_len", type=int, default=20480)
parser.add_argument("--gpu_mem_gib", type=float, default=0.0,
                    help="GPU memory budget for vLLM in GiB (0: 90%% of the card). h100-47 slots share one H100 NVL "
                         "with another job, so 90%% of the card is not free there (job 1164403)")
parser.add_argument("--limit", type=int, default=0, help="Use only the first N questions (smoke test)")
parser.add_argument("--seed", type=int, default=20261119)
parser.add_argument("--prompt", choices=["full", "colour"], default="full",
                    help="full: cube semantics, answers up + above_red. colour: ablation without the two-part cube "
                         "semantics, asks only for the colour covering most of the upward face (scored against up)")
args = parser.parse_args()

COLOURS = ["white", "magenta", "red", "green", "yellow", "blue"]
CAMERA_TEXT = {
    "c1": "camera C1, in front of the hand and above it, looking down at about 38 degrees",
    "s040": "camera S040, beside the hand, looking slightly down (about 30 degrees)",
    "s130": "camera S130, beside the hand, looking slightly down (about 30 degrees)",
    "s220": "camera S220, beside the hand on the opposite side from S040, looking slightly down (about 30 degrees)",
    "top": "camera TOP, above the hand, looking almost straight down",
}
INTRO = """You are looking at a cube held by a robot hand (palm up).

The cube has two parts: a bottom part (two thirds of its height) and a top layer (the remaining third) that can \
turn about the cube's axis in quarter turns, like one layer of a Rubik's cube. Dark seams separate the faces and \
the two parts.
Colours:
- The four sides of the bottom part are red, green, yellow and blue, in that order going counter-clockwise when \
the cube is seen from the top layer's side.
- The bottom face (opposite the top layer) is magenta.
- The top layer's outer face is white with a short black stripe.
- The top layer's four sides use the same four colours. When the top layer is aligned (not turned), each side of \
the cube shows a single colour from bottom to top; after a quarter turn the top layer's colours are shifted by one \
side.
The hand may hide parts of the cube. The images below show the same moment from fixed cameras."""
QUESTION = """Answer two questions about the cube in these images:
1. up: which face of the cube points up (toward the sky)? Answer "white" if the top layer's white face points up, \
"magenta" if the bottom face points up, otherwise the colour of the bottom part on the side that points up.
2. above_red: on the bottom part's red side, which colour does the top layer show directly above it? ("red" if \
the top layer is aligned.)
Colours: white, magenta, red, green, yellow, blue.
Think as needed, then end your reply with one line of JSON only, for example {"up": "white", "above_red": "blue"}"""
# Ablation: no layer / bottom-part semantics. For a side pointing up, the bottom part's colour covers two thirds of
# the upward face (the top layer's strip one third), so "the colour covering most of it" equals the up answer.
COLOUR_INTRO = """You are looking at a cube held by a robot hand (palm up). Each face of the cube is coloured; a \
face may show more than one colour. The hand may hide parts of the cube. The images below show the same moment \
from fixed cameras."""
COLOUR_QUESTION = """Which colour covers most of the cube face that points up (toward the sky)?
Colours: white, magenta, red, green, yellow, blue.
Think as needed, then end your reply with one line of JSON only, in the form {"up": "<colour>"}"""
KEYS = ("up", "above_red") if args.prompt == "full" else ("up",)
if args.prompt == "colour":
    INTRO, QUESTION = COLOUR_INTRO, COLOUR_QUESTION


def parse_answer(text: str) -> dict | None:
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    for match in reversed(re.findall(r"\{[^{}]*\}", text)):
        try:
            obj = json.loads(match)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and all(k in obj for k in KEYS):
            return {k: str(obj[k]).strip().lower() for k in KEYS}
    return None


def image_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def build_messages(question: dict, cams: list[str], nohand: bool) -> list[dict]:
    content = [{"type": "text", "text": INTRO}]
    run_dir = Path(question["run_dir"])
    for cam in cams:
        suffix = "_nohand" if nohand else ""
        path = run_dir / "images" / f"s{question['step']:04d}_e{question['env']:03d}_{cam}{suffix}.png"
        content.append({"type": "text", "text": f"Image from {CAMERA_TEXT[cam]}:"})
        content.append({"type": "image_url", "image_url": {"url": image_url(path)}})
    content.append({"type": "text", "text": QUESTION})
    return [{"role": "user", "content": content}]


def main() -> None:
    from vllm import LLM, SamplingParams

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    questions = [json.loads(line) for line in open(args.questions, encoding="utf-8")]
    if args.limit:
        questions = questions[: args.limit]
    conditions = []
    for spec in args.conditions.split(","):
        cams_spec, _, flag = spec.partition("@")
        conditions.append((spec, cams_spec.split("+"), flag == "nohand"))
    max_images = max(len(c[1]) for c in conditions)

    import torch

    free, total = torch.cuda.mem_get_info()
    utilization = 0.9 if args.gpu_mem_gib <= 0 else args.gpu_mem_gib * 2**30 / total
    print(f"gpu_free_gib={free / 2**30:.1f} gpu_total_gib={total / 2**30:.1f} vllm_utilization={utilization:.3f}",
          flush=True)
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": max_images},
        gpu_memory_utilization=utilization,
        seed=args.seed,
    )
    params = SamplingParams(n=args.samples, temperature=args.temperature, top_p=1.0,
                            max_tokens=args.max_tokens, seed=args.seed)

    responses = open(out / "responses.jsonl", "w", encoding="utf-8")
    summary = {"model": args.model, "gpu_memory_utilization": utilization, "questions": args.questions, "num_questions": len(questions),
               "samples": args.samples, "temperature": args.temperature, "max_tokens": args.max_tokens,
               "prompt": args.prompt, "intro": INTRO, "question": QUESTION, "conditions": {}}
    for spec, cams, nohand in conditions:
        t0 = time.time()
        conversations = [build_messages(q, cams, nohand) for q in questions]
        outputs = llm.chat(conversations, params, use_tqdm=False)
        elapsed = time.time() - t0
        tally = collections.Counter()
        strata = collections.defaultdict(lambda: [0, 0])
        informative = 0
        out_tokens = 0
        for q, result in zip(questions, outputs):
            vis = [q["visible_fraction"].get(c) or 0.0 for c in cams]
            vis_bin = "vis<0.8" if min(vis) < 0.8 else ("vis<0.9" if min(vis) < 0.9 else "vis>=0.9")
            correct_both = 0
            for index, completion in enumerate(result.outputs):
                answer = parse_answer(completion.text)
                up_ok = bool(answer) and answer["up"] == q["answer"]["up"]
                red_ok = bool(answer) and "above_red" in KEYS and answer["above_red"] == q["answer"]["above_red"]
                both = up_ok and (red_ok or "above_red" not in KEYS)
                correct_both += both
                out_tokens += len(completion.token_ids)
                tally["completions"] += 1
                tally["parse_fail"] += answer is None
                tally["up"] += up_ok
                tally["above_red"] += red_ok
                tally["both"] += both
                tally["truncated"] += completion.finish_reason == "length"
                for key in (f"up={q['answer']['up']}", f"detent={q['detent']}", vis_bin):
                    strata[key][0] += both
                    strata[key][1] += 1
                responses.write(json.dumps({
                    "id": q["id"], "condition": spec, "sample": index, "answer": answer, "truth": q["answer"],
                    "up_ok": up_ok, "above_red_ok": red_ok, "tokens": len(completion.token_ids),
                    "finish": completion.finish_reason, "text": completion.text,
                }) + "\n")
            informative += 0 < correct_both < len(result.outputs)
        n = max(tally["completions"], 1)
        record = {
            "acc_up": tally["up"] / n,
            "acc_above_red": tally["above_red"] / n,
            "acc_both": tally["both"] / n,
            "parse_fail": tally["parse_fail"] / n,
            "truncated": tally["truncated"] / n,
            "informative_group_rate": informative / max(len(questions), 1),
            "mean_output_tokens": out_tokens / n,
            "seconds": elapsed,
            "strata_acc_both": {k: [v[0] / v[1], v[1]] for k, v in sorted(strata.items())},
        }
        summary["conditions"][spec] = record
        print(f"z1_condition={spec} " + json.dumps({k: round(v, 3) for k, v in record.items()
                                                    if isinstance(v, float)}), flush=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    responses.close()
    print("z1_vlm_eval_status=PASS")


if __name__ == "__main__":
    main()
