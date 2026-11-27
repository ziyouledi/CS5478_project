#!/usr/bin/env python3
"""Zero-extra-dependency PyTorch LoRA SFT trainer for Qwen3-VL-8B-Instruct (Scheme C & Smart-Cube Fallback).

Uses only `torch` and `transformers` already installed in `~/software/vlm-stack/env_vllm` (no `peft` / `trl`
required). Injects standard LoRA adapters into all attention & MLP linear layers of the language model plus
the vision-language merger/projector (keeping the ViT backbone frozen to preserve general visual features).

After training finishes:
  1. Saves the lightweight LoRA adapter checkpoint (`lora_adapter.pt`).
  2. Merges LoRA weights back into the base bf16 model and saves a standard HuggingFace model directory
     (`<output_dir>/merged_hf/`) so `vllm` (`z1_mcq_eval.py`, `bench_vlm_eval.py`, and closed-loop RL)
     can load it directly with zero code changes!
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


class LoRALinear(nn.Module):
    """Drop-in LoRA wrapper around an existing nn.Linear layer."""

    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 32.0, dropout: float = 0.05):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.r = r
        self.scaling = alpha / float(r)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        device = base.weight.device
        dtype = base.weight.dtype
        self.lora_A = nn.Parameter(torch.empty((r, base.in_features), device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros((base.out_features, r), device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return out + delta * self.scaling

    @torch.no_grad()
    def merge_to_base(self) -> nn.Linear:
        delta = (self.lora_B @ self.lora_A) * self.scaling
        self.base.weight.add_(delta.to(self.base.weight.dtype))
        return self.base


def inject_lora(model: nn.Module, target_suffixes: tuple[str, ...], r: int, alpha: float, dropout: float) -> list[str]:
    """Freeze all parameters, then wrap matching linear modules (excluding ViT blocks) with LoRALinear."""
    for p in model.parameters():
        p.requires_grad_(False)

    replaced = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        # Freeze vision encoder blocks; allow language_model + vision merger/projector
        if "visual.blocks." in name or "vision_tower" in name:
            continue
        if not any(name.endswith(s) for s in target_suffixes) and "merger" not in name:
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
        replaced.append(name)
    return replaced


def merge_and_unwrap_lora(model: nn.Module) -> None:
    """Merge all LoRALinear deltas into their base nn.Linear and restore clean HF module tree."""
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            parent_name, _, child_name = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, module.merge_to_base())


def prepare_sample(processor, sample: dict, max_seq_len: int) -> dict | None:
    """Tokenize one (image, system, user, assistant) sample and mask prompt tokens with -100."""
    img = Image.open(sample["image"]).convert("RGB")
    prompt_msgs = []
    if sample.get("system"):
        prompt_msgs.append({"role": "system", "content": sample["system"]})
    prompt_msgs.append({
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": sample["user"]},
        ],
    })
    full_msgs = prompt_msgs + [{"role": "assistant", "content": sample["assistant"]}]

    prompt_text = processor.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)

    prompt_inputs = processor(text=[prompt_text], images=[img], return_tensors="pt")
    full_inputs = processor(text=[full_text], images=[img], return_tensors="pt")

    seq_len = full_inputs["input_ids"].shape[1]
    if seq_len > max_seq_len:
        return None
    prompt_len = prompt_inputs["input_ids"].shape[1]
    labels = full_inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100
    full_inputs["labels"] = labels
    return full_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local HF path to Qwen3-VL-8B-Instruct")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--max_samples", type=int, default=0, help="Optional cap for fast smoke tests")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in open(args.train_jsonl, encoding="utf-8")]
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    print(f"Loaded {len(rows)} SFT samples from {args.train_jsonl}", flush=True)

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
    ).to("cuda:0")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    target_suffixes = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    replaced = inject_lora(model, target_suffixes, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"LoRA injected into {len(replaced)} modules; trainable={n_train:,} / {n_total:,} ({100*n_train/n_total:.2f}%)",
          flush=True)

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    total_opt_steps = max(1, (len(rows) * args.epochs) // args.grad_accum)
    warmup_steps = max(1, int(0.05 * total_opt_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_opt_steps - warmup_steps)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    t0 = time.time()
    model.train()
    opt_step = 0
    accum_loss = 0.0
    micro_in_accum = 0
    skipped = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(len(rows)).tolist()
        for step_idx, r_idx in enumerate(perm, start=1):
            batch = prepare_sample(processor, rows[r_idx], args.max_seq_len)
            if batch is None:
                skipped += 1
                continue
            batch = {k: v.to("cuda:0") for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / float(args.grad_accum)
            loss.backward()
            accum_loss += float(outputs.loss.item())
            micro_in_accum += 1

            if micro_in_accum >= args.grad_accum or step_idx == len(perm):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1
                avg_loss = accum_loss / max(micro_in_accum, 1)
                accum_loss = 0.0
                micro_in_accum = 0
                if opt_step % args.log_every == 0 or opt_step == 1 or step_idx == len(perm):
                    lr_now = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    rec = {"epoch": epoch, "opt_step": opt_step, "loss": round(avg_loss, 4),
                           "lr": lr_now, "elapsed_s": round(elapsed, 1)}
                    history.append(rec)
                    print(f"sft_step={opt_step}/{total_opt_steps} epoch={epoch} loss={avg_loss:.4f} "
                          f"lr={lr_now:.2e} elapsed={elapsed:.0f}s", flush=True)

    # Save LoRA adapter weights
    adapter_state = {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}
    torch.save({"args": vars(args), "modules": replaced, "state_dict": adapter_state}, out / "lora_adapter.pt")
    (out / "train_history.json").write_text(json.dumps({
        "samples": len(rows), "skipped": skipped, "epochs": args.epochs, "opt_steps": opt_step,
        "elapsed_s": time.time() - t0, "history": history,
    }, indent=2) + "\n", encoding="utf-8")

    # Merge LoRA into base model and save HF checkpoint for direct vLLM evaluation
    print("Merging LoRA weights into base model for vLLM...", flush=True)
    merge_and_unwrap_lora(model)
    merged_dir = out / "merged_hf"
    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged_dir, safe_serialization=True)
    processor.save_pretrained(merged_dir)
    print(f"Saved merged HF model to {merged_dir}", flush=True)
    print("train_qwen3vl_lora_sft_status=PASS", flush=True)


if __name__ == "__main__":
    main()
