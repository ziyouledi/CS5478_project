#!/usr/bin/env python3
"""Stage C2: Multi-turn Visual GRPO trainer for Qwen3-VL-8B-Instruct (warm-started from C1 vision_full SFT).

Zero extra dependencies (pure PyTorch + transformers in `env_vllm`). Uses the pre-built 82-state real
hand-occluded 3-view `montage` image pool (`state_image_pool.json`) for millisecond-speed visual transitions,
with strict exclusion of the 200 frozen test (src, dst) pairs.

Supports two reward ablations (PLAN_A_VS_C.md):
  - `grounded` (C2-Grounded): R = R_ep + lambda_state * mean_t I(pred_state_t == true_state_t)
  - `terminal` (C2-Terminal): R = R_ep only (lambda_state = 0.0)
where R_ep = 1.0 - 0.05 * max(0, rounds - L) on reaching dst, -0.2 on JSON parse failure, else 0.0.

Uses exact group-normalized GRPO advantage with DAPO zero-variance filtering and exact KL penalty against
the frozen SFT reference (computed with zero extra memory by temporarily disabling LoRA scaling).
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from build_c1_sft_dataset import SYS_VISION_FULL, on_red_strip
from cube_state_graph import DIST_TABLE, ORIENTATIONS, index_to_state, state_to_index
from eval_c1_openloop_plan import apply_action, extract_json, parse_action, parse_pred_state
from train_qwen3vl_lora_sft import LoRALinear, inject_lora, merge_and_unwrap_lora


@contextlib.contextmanager
def disable_lora(model: torch.nn.Module):
    saved = []
    for m in model.modules():
        if isinstance(m, LoRALinear):
            saved.append((m, m.scaling))
            m.scaling = 0.0
    try:
        yield
    finally:
        for m, s in saved:
            m.scaling = s


def build_prompt_text(processor, dst_state: dict) -> str:
    msgs = [
        {"role": "system", "content": SYS_VISION_FULL},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": (
                f"Target State: {json.dumps(dst_state)}\n"
                "Observe the 3-camera montage, output `view_obs`, `state`, `thought`, and the next optimal `action` in JSON."
            )},
        ]},
    ]
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def compute_completion_logprobs(model, processor, img: Image.Image, prompt_text: str, comp_text: str) -> torch.Tensor:
    """Return per-token log-probabilities for `comp_text` tokens (shape [T_comp])."""
    prompt_inputs = processor(text=[prompt_text], images=[img], return_tensors="pt")
    full_inputs = processor(text=[prompt_text + comp_text], images=[img], return_tensors="pt")
    prompt_len = prompt_inputs["input_ids"].shape[1]
    full_len = full_inputs["input_ids"].shape[1]
    if full_len <= prompt_len:
        return torch.zeros(0, device="cuda:0")
    full_inputs = {k: v.to("cuda:0") for k, v in full_inputs.items()}
    logits = model(**full_inputs).logits[0]  # (full_len, V)
    # Token at position t is predicted by logits[t-1]
    target_ids = full_inputs["input_ids"][0, prompt_len:full_len]
    pred_logits = logits[prompt_len - 1 : full_len - 1]
    log_probs = F.log_softmax(pred_logits.float(), dim=-1)
    token_lp = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return token_lp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="C1 SFT merged_hf checkpoint")
    parser.add_argument("--train_pool", required=True, help="state_image_pool.json")
    parser.add_argument("--exclude_tasks", required=True, help="Frozen 200 test tasks.json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--reward_mode", required=True, choices=["grounded", "terminal"])
    parser.add_argument("--lambda_state", type=float, default=0.25)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--tasks_per_step", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--kl_beta", type=float, default=0.02)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--max_new_tokens", type=int, default=320)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    raw_pool = json.loads(Path(args.train_pool).read_text(encoding="utf-8"))
    pool: dict[int, list[str]] = {int(k): v for k, v in raw_pool.items() if v}
    covered = sorted(pool.keys())
    covered_set = set(covered)

    # Pre-load PIL images into RAM (656 small PNGs ≈ 400 MB) for fast zero-IO rollouts
    img_cache: dict[str, Image.Image] = {}
    for paths in pool.values():
        for p in paths:
            if p not in img_cache:
                img_cache[p] = Image.open(p).convert("RGB")

    # Exclude all 200 frozen test (src, dst) pairs
    excl = set()
    for t in json.loads(Path(args.exclude_tasks).read_text(encoding="utf-8")):
        s = state_to_index(int(t["src"]["orient_id"]), int(t["src"]["detent"]))
        d = state_to_index(int(t["dst"]["orient_id"]), int(t["dst"]["detent"]))
        excl.add((s, d))

    # Build candidate training pairs grouped by L in {1, 2, 3, 4} where src is in covered_set
    pairs_by_L: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    for s in covered:
        for d in covered:
            if s == d or (s, d) in excl:
                continue
            L = int(DIST_TABLE[s, d])
            if 1 <= L <= 4:
                pairs_by_L[L].append((s, d))
    print("train_pairs_by_L:", {L: len(v) for L, v in sorted(pairs_by_L.items())}, flush=True)

    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda:0")
    target_suffixes = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    replaced = inject_lora(model, target_suffixes, r=args.lora_r, alpha=args.lora_alpha, dropout=0.0)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"C2 GRPO mode={args.reward_mode} LoRA modules={len(replaced)} trainable={sum(p.numel() for p in trainable):,}", flush=True)

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    L_weights = [1, 2, 2, 2, 3, 3, 3, 4, 4]  # focus on L=2,3,4 where turn & multi-step errors happen

    t0 = time.time()
    history = []
    effective_groups = 0
    total_groups = 0

    for step in range(1, args.steps + 1):
        # --- Phase 1: Rollout collection (eval mode, KV cache enabled) ---
        model.eval()
        model.config.use_cache = True
        step_trajs = []  # list of (advantage, list of (img, prompt_text, comp_text))
        step_rewards = []
        step_succ = []
        step_state_acc = []

        for _ in range(args.tasks_per_step):
            L = rng.choice(L_weights)
            src_idx, dst_idx = rng.choice(pairs_by_L[L])
            src = index_to_state(src_idx)
            dst = index_to_state(dst_idx)
            dst_state = {
                "sky_face": ORIENTATIONS[dst[0]]["up_colour"],
                "front_face": ORIENTATIONS[dst[0]]["front_colour"],
                "on_red_panel_strip": on_red_strip(dst[1]),
            }
            prompt_text = build_prompt_text(processor, dst_state)
            max_rounds = min(L + 2, 6)

            # Run G parallel episodes for this (src, dst) task
            cur_list = [src] * args.group_size
            done_list = [False] * args.group_size
            succ_list = [False] * args.group_size
            parse_fail_list = [False] * args.group_size
            rounds_list = [0] * args.group_size
            state_hits = [[] for _ in range(args.group_size)]
            turn_records = [[] for _ in range(args.group_size)]

            for _rnd in range(1, max_rounds + 1):
                active_g = [g for g in range(args.group_size) if not done_list[g]]
                if not active_g:
                    break
                batch_imgs = []
                batch_texts = []
                valid_g = []
                for g in active_g:
                    sidx = state_to_index(*cur_list[g])
                    if sidx not in covered_set:
                        done_list[g] = True
                        continue
                    valid_g.append(g)
                    batch_imgs.append(img_cache[rng.choice(pool[sidx])])
                    batch_texts.append(prompt_text)
                if not valid_g:
                    break

                inputs = processor(text=batch_texts, images=batch_imgs, padding=True, return_tensors="pt").to("cuda:0")
                in_len = inputs["input_ids"].shape[1]
                with torch.no_grad():
                    gen_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=0.95,
                        pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
                    )
                for idx_b, g in enumerate(valid_g):
                    comp_ids = gen_ids[idx_b, in_len:]
                    comp_text = processor.tokenizer.decode(comp_ids, skip_special_tokens=True)
                    turn_records[g].append((batch_imgs[idx_b], prompt_text, comp_text))
                    rounds_list[g] += 1

                    obj = extract_json(comp_text)
                    valid_json = isinstance(obj, dict)
                    pred_st = parse_pred_state(obj.get("state") if valid_json else None)
                    pred_act = parse_action(obj.get("action") if valid_json else None)
                    state_hits[g].append(1.0 if pred_st == cur_list[g] else 0.0)

                    if pred_act is None:
                        parse_fail_list[g] = True
                    else:
                        nxt = apply_action(cur_list[g], pred_act)
                        if nxt is not None:
                            cur_list[g] = nxt
                    if cur_list[g] == dst:
                        succ_list[g] = True
                        done_list[g] = True
                    elif rounds_list[g] >= max_rounds:
                        done_list[g] = True

            # Compute group rewards
            rewards = []
            for g in range(args.group_size):
                if succ_list[g]:
                    r_ep = 1.0 - 0.05 * max(0, rounds_list[g] - L)
                elif parse_fail_list[g]:
                    r_ep = -0.2
                else:
                    r_ep = 0.0
                r_st = float(np.mean(state_hits[g])) if state_hits[g] else 0.0
                r_tot = r_ep + (args.lambda_state * r_st if args.reward_mode == "grounded" else 0.0)
                rewards.append(r_tot)
                step_rewards.append(r_tot)
                step_succ.append(1.0 if succ_list[g] else 0.0)
                step_state_acc.append(r_st)

            total_groups += 1
            r_arr = np.asarray(rewards, dtype=np.float64)
            r_std = float(r_arr.std())
            if r_std < 1e-6:
                continue  # DAPO dynamic filter: skip zero-variance group
            effective_groups += 1
            advs = (r_arr - float(r_arr.mean())) / (r_std + 1e-6)
            for g in range(args.group_size):
                if abs(float(advs[g])) > 1e-5 and turn_records[g]:
                    step_trajs.append((float(advs[g]), turn_records[g]))

        # --- Phase 2: GRPO Policy Gradient + KL update ---
        loss_val = 0.0
        kl_val = 0.0
        n_turns_updated = 0
        if step_trajs:
            model.train()
            model.config.use_cache = False
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            optimizer.zero_grad(set_to_none=True)
            total_turns = sum(len(tr) for _, tr in step_trajs)
            for adv, tr in step_trajs:
                for img, p_txt, c_txt in tr:
                    with torch.no_grad(), disable_lora(model):
                        ref_lp = compute_completion_logprobs(model, processor, img, p_txt, c_txt)
                    if ref_lp.numel() == 0:
                        continue
                    cur_lp = compute_completion_logprobs(model, processor, img, p_txt, c_txt)
                    # Unbiased Schulman KL approx: exp(ref - cur) - (ref - cur) - 1
                    diff = (ref_lp - cur_lp).clamp(-10.0, 10.0)
                    kl = (torch.exp(diff) - diff - 1.0).mean()
                    # GRPO surrogate (single on-policy step -> ratio=1, grad = -adv * grad(log pi))
                    pg_loss = -(adv * cur_lp.mean()) + args.kl_beta * kl
                    (pg_loss / float(total_turns)).backward()
                    loss_val += float(pg_loss.item())
                    kl_val += float(kl.item())
                    n_turns_updated += 1
            if n_turns_updated > 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                loss_val /= n_turns_updated
                kl_val /= n_turns_updated
            model.gradient_checkpointing_disable()

        rec = {
            "step": step,
            "reward_mean": round(float(np.mean(step_rewards)), 4) if step_rewards else 0.0,
            "succ_rate": round(float(np.mean(step_succ)), 4) if step_succ else 0.0,
            "state_acc": round(float(np.mean(step_state_acc)), 4) if step_state_acc else 0.0,
            "eff_group_frac": round(effective_groups / max(total_groups, 1), 3),
            "turns_updated": n_turns_updated,
            "loss": round(loss_val, 4),
            "kl": round(kl_val, 4),
            "elapsed_s": round(time.time() - t0, 1),
        }
        history.append(rec)
        print(
            f"grpo_step={step}/{args.steps} succ={rec['succ_rate']:.3f} st_acc={rec['state_acc']:.3f} "
            f"rew={rec['reward_mean']:.3f} eff_g={rec['eff_group_frac']:.2f} kl={rec['kl']:.4f} "
            f"turns={n_turns_updated} elapsed={rec['elapsed_s']:.0f}s",
            flush=True,
        )

    # Save LoRA adapter & merged HF checkpoint
    adapter_state = {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}
    torch.save({"args": vars(args), "modules": replaced, "state_dict": adapter_state}, out / "lora_adapter.pt")
    (out / "train_history.json").write_text(json.dumps({
        "args": vars(args), "effective_groups": effective_groups, "total_groups": total_groups,
        "elapsed_s": round(time.time() - t0, 1), "history": history,
    }, indent=2) + "\n", encoding="utf-8")

    print("Merging C2 GRPO LoRA weights into merged_hf...", flush=True)
    merge_and_unwrap_lora(model)
    merged_dir = out / "merged_hf"
    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged_dir, safe_serialization=True)
    processor.save_pretrained(merged_dir)
    print(f"Saved merged HF model to {merged_dir}", flush=True)
    print("train_c2_visual_grpo_status=PASS", flush=True)


if __name__ == "__main__":
    main()
