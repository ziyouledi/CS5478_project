"""Shared multiple-choice helpers for bench_vlm_eval.py and z1_mcq_eval.py, so Z1 uses exactly the template and
answer extraction that reproduced the public benchmarks (job 1166604).

Prompt: the VLMEvalKit ImageMCQ format (also the Qwen3-VL report's template for MMBench / MMStar / BLINK):
    [Hint: {hint}\n]Question: {question}\nOptions:\nA. ...\nB. ...\nPlease select the correct answer from the options above. \n
"""

from __future__ import annotations

import base64
import re

INSTRUCTION = "Please select the correct answer from the options above. \n"
LETTERS = "ABCDEFGHIJ"


def image_url(data: bytes) -> str:
    mime = "image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def build_prompt(question: str, options: dict, hint: str = "") -> str:
    text = f"Hint: {hint}\n" if hint else ""
    text += f"Question: {question}\n"
    if options:
        text += "Options:\n" + "".join(f"{k}. {v}\n" for k, v in options.items())
    return text + INSTRUCTION


def extract(text: str, options: dict) -> str | None:
    """Rule-based letter extraction; None if no unique answer is found (scored as wrong)."""
    text = text.rsplit("</think>", 1)[-1].strip()
    keys = set(options) or set("ABCD")
    found = [m for m in re.findall(r"answer(?: is)?\s*[:：]?\s*\**\(?([A-J])\)?\b", text, re.I) if m in keys]
    if found:
        return found[-1]
    match = re.match(r"^\(?([A-J])\)?(?:[.:)]|\s*$)", text)  # "B", "(B)", "B. ..." (not the article "A cat")
    if match and match.group(1) in keys:
        return match.group(1)
    standalone = {m for m in re.findall(r"(?<![A-Za-z])\(?([A-J])\)?(?![A-Za-z])", text) if m in keys}
    if len(standalone) == 1:
        return standalone.pop()
    hits = [k for k, v in options.items() if str(v).strip() and str(v).strip().lower() in text.lower()]
    return hits[0] if len(hits) == 1 else None
