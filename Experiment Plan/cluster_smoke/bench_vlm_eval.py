#!/usr/bin/env python3
"""Pipeline check for the Z1 VLM evaluation (GOAL_LEVEL_RL_PLAN.md, Z1 评测规范 step 1).

Runs public multiple-choice benchmarks through the same vLLM path as z1_vlm_eval.py (LLM.chat, base64 image_url,
images before text) and compares accuracy with the numbers the Qwen team reports. If they match, the chat template
and image passing are sound; if not, the Z1 results are suspect too.

Prompt: the VLMEvalKit ImageMCQ format, which is also the Qwen3-VL report's template for MMBench / MMStar / BLINK
(Appendix B.2, B.7):
    Question: {question}\nOptions:\nA. ...\nB. ...\nPlease select the correct answer from the options above. \n
Decoding: "greedy" (temperature 0, the benchmark convention) and "qwen" (the Qwen3-VL README Instruct settings:
temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5, seed 3407).
Scoring: rule-based letter extraction (no LLM judge, so unmatched answers count as wrong; the unmatched rate is
reported).

Benchmarks (parquet files from the Hugging Face hub, downloaded by 36_bench_prep.sbatch):
  blink       BLINK-Benchmark/BLINK, val split of all 14 subtasks (1901 questions, 1-4 images)
  mmstar      Lin-Chen/MMStar (1500 questions)
  embspatial  FlagEval/EmbSpatial-Bench (test split)
"""

from __future__ import annotations

import argparse
import base64
import collections
import glob
import json
import re
import time
from pathlib import Path

from mcq_common import INSTRUCTION, LETTERS, build_prompt, extract, image_url

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", required=True)
parser.add_argument("--data_dir", required=True, help="Directory with blink/, mmstar/, embspatial/ snapshots")
parser.add_argument("--output_dir", required=True)
parser.add_argument("--benchmarks", default="blink,mmstar,embspatial")
parser.add_argument("--decodings", default="greedy,qwen")
parser.add_argument("--max_tokens", type=int, default=4096)
parser.add_argument("--max_model_len", type=int, default=32768)
parser.add_argument("--gpu_mem_gib", type=float, default=0.0, help="0: 90%% of the card")
parser.add_argument("--chunk", type=int, default=200,
                    help="Questions per llm.chat call; all of BLINK at once ran the host out of memory (job 1166173)")
parser.add_argument("--limit", type=int, default=0, help="First N questions per benchmark (smoke test)")
args = parser.parse_args()

# Qwen3-VL-8B-Instruct, official table (qwen3vl_4b_8b_vl_instruct.jpg in the Hugging Face model card).
REPORTED = {"blink": 69.1, "mmstar": 70.9, "embspatial": 78.5}
def image_bytes(value) -> bytes:
    """Parquet image cells are either raw bytes or the datasets Image struct {"bytes": ..., "path": ...}."""
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, str):
        return base64.b64decode(value)
    return value["bytes"]


def read_parquet(pattern: str) -> list[dict]:
    import pyarrow.parquet as pq

    rows = []
    for path in sorted(glob.glob(pattern)):
        rows += pq.read_table(path).to_pylist()
    if not rows:
        raise FileNotFoundError(pattern)
    return rows


def load_blink() -> list[dict]:
    items = []
    for row in read_parquet(f"{args.data_dir}/blink/*/val-*.parquet"):
        images = [image_bytes(row[f"image_{i}"]) for i in range(1, 5) if row.get(f"image_{i}")]
        items.append({"id": row["idx"], "group": row["sub_task"], "images": images, "question": row["question"],
                      "options": dict(zip(LETTERS, row["choices"])), "answer": row["answer"].strip("()")})
    return items


def load_mmstar() -> list[dict]:
    items = []
    for row in read_parquet(f"{args.data_dir}/mmstar/*.parquet"):
        question, _, opts = row["question"].partition("\nOptions: ")
        options = dict(re.findall(r"([A-D]): (.*?)(?:, (?=[A-D]: )|$)", opts.strip(), re.S))
        if len(options) < 2:  # keep the raw text if the options cannot be split
            question, options = row["question"], {}
        items.append({"id": row["index"], "group": row["category"], "images": [image_bytes(row["image"])],
                      "question": question, "options": options, "answer": row["answer"].strip()})
    return items


def load_embspatial() -> list[dict]:
    items = []
    for row in read_parquet(f"{args.data_dir}/embspatial/data/*.parquet"):
        data = image_bytes(row["image"]) if row.get("image") else base64.b64decode(row["image_base64"])
        items.append({"id": row["question_id"], "group": row["relation"], "images": [data],
                      "question": row["question"], "options": dict(zip(LETTERS, row["answer_options"])),
                      "answer": LETTERS[int(row["answer"])]})
    return items


def item_prompt(item: dict) -> str:
    return build_prompt(item["question"], item["options"])


def main() -> None:
    import torch
    from vllm import LLM, SamplingParams

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    loaders = {"blink": load_blink, "mmstar": load_mmstar, "embspatial": load_embspatial}
    data = {}
    for name in args.benchmarks.split(","):
        items = loaders[name]()
        data[name] = items[: args.limit] if args.limit else items
        print(f"bench_loaded={name} questions={len(data[name])}", flush=True)
    max_images = max(len(i["images"]) for items in data.values() for i in items)

    free, total = torch.cuda.mem_get_info()
    utilization = 0.9 if args.gpu_mem_gib <= 0 else args.gpu_mem_gib * 2**30 / total
    print(f"gpu_free_gib={free / 2**30:.1f} gpu_total_gib={total / 2**30:.1f} vllm_utilization={utilization:.3f}",
          flush=True)
    llm = LLM(model=args.model, max_model_len=args.max_model_len, limit_mm_per_prompt={"image": max_images},
              gpu_memory_utilization=utilization, seed=3407)
    decodings = {
        "greedy": SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
        "qwen": SamplingParams(temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5,
                               max_tokens=args.max_tokens, seed=3407),
    }

    summary = {"model": args.model, "reported": REPORTED, "instruction": INSTRUCTION, "results": {}}
    responses = open(out / "responses.jsonl", "w", encoding="utf-8")
    for name, items in data.items():
        def conversation(item: dict) -> list[dict]:
            content = [{"type": "image_url", "image_url": {"url": image_url(b)}} for b in item["images"]]
            content.append({"type": "text", "text": item_prompt(item)})
            return [{"role": "user", "content": content}]

        for decoding in args.decodings.split(","):
            t0 = time.time()
            outputs = []
            for start in range(0, len(items), args.chunk):
                chunk = [conversation(item) for item in items[start:start + args.chunk]]
                outputs += llm.chat(chunk, decodings[decoding], use_tqdm=False)
            tally = collections.Counter()
            groups = collections.defaultdict(lambda: [0, 0])
            for item, result in zip(items, outputs):
                completion = result.outputs[0]
                pred = extract(completion.text, item["options"])
                ok = pred == item["answer"]
                tally["n"] += 1
                tally["correct"] += ok
                tally["unmatched"] += pred is None
                tally["truncated"] += completion.finish_reason == "length"
                tally["tokens"] += len(completion.token_ids)
                groups[item["group"]][0] += ok
                groups[item["group"]][1] += 1
                responses.write(json.dumps({"bench": name, "decoding": decoding, "id": str(item["id"]),
                                            "group": item["group"], "answer": item["answer"], "pred": pred,
                                            "ok": ok, "finish": completion.finish_reason,
                                            "text": completion.text}) + "\n")
            n = max(tally["n"], 1)
            record = {"acc": 100 * tally["correct"] / n, "reported": REPORTED.get(name),
                      "unmatched": tally["unmatched"] / n, "truncated": tally["truncated"] / n,
                      "mean_tokens": tally["tokens"] / n, "n": tally["n"], "seconds": time.time() - t0,
                      "groups": {k: [100 * a / b, b] for k, (a, b) in sorted(groups.items())}}
            summary["results"][f"{name}/{decoding}"] = record
            print(f"bench_result={name}/{decoding} " + json.dumps(
                {k: round(v, 3) if isinstance(v, float) else v for k, v in record.items() if k != "groups"}),
                flush=True)
            (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    responses.close()
    print("bench_vlm_eval_status=PASS")


if __name__ == "__main__":
    main()
