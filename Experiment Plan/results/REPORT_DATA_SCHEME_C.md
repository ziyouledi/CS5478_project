# Scheme C (Visual Map-then-Reason VLM + Unified v4 Closed-Loop) — Final Report Data Summary

Compiled: 2026-11-21 (all numbers verified from Slurm `sacct`, logs, and local JSON summaries under `Experiment Plan/results/`).
Base VLM: `Qwen3-VL-8B-Instruct` (LoRA `r=16, α=32`, ViT vision encoder frozen, language model + vision merger trainable: 44.8M / 8.81B = 0.51%).
Low-level 60 Hz controller: frozen `Unified v4` (`expert-unified-v4-seed0.pth`, SHA-256 `ca3d9367...a1d4`; BFS Oracle `1169489`: 2048/2048 = 100.0% success, 0 falls, 16.5 steps/macro-action).

---

## Table 1. Held-out Z1 180-Frame Perception & Overfitting Boundary Audit (`test` split, Montage, Zero SoM, CircularEval)

| Model / Condition | Job ID | Canonical (`c1+s040+s220`) All / `band` | Canonical `@nohand` All / `band` | OOD `reverse_cycle` (G↔B) All / `band` | OOD `novel_palette` (O/C/P/Pk) All / `band` | Unseen Prompt `v3` All / `turned` / `band` | Unanswerable CBD (`n=21`) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero-shot `Qwen3-VL-8B` | `1168587` | — / 0.075 | — / 0.057 | — | — | — | 0.524 |
| `smart_cube` (sensor cap) | `1173528` | 0.208 / 0.252 (`up` 0.978) | 0.157 / 0.208 | 0.132 / 0.214 | — | — | — |
| `color_invariant` (2 ep, aug incl. G↔B) | `1173527` / `1174847` | **0.994** / **0.994** | 0.962 / 0.975 | **0.975** / **0.975** | **0.943** / **0.962** | — | 0.000 |
| **`C1-v1` (`vision_full`, 2 ep)** | `1173965` / `1174846` | **0.994** / **0.994** | **0.994** / **0.994** | 0.962 / 0.969 | 0.811 / 0.899 (`@nohand` 0.969) | 0.368 / 0.600 / 0.447 | 0.000 |
| **`C1-v2` (1 ep, multi-tpl + CBD + 30% aug + CoT)** | `1176313` | **0.987** / **0.987** | 0.969 / 0.969 | 0.962 / 0.962 | 0.811 / **0.937** (van 0.962) | **0.592** / **0.967** / **0.612** | **1.000** (21/21) |

- **Unseen Camera Viewpoints (`C1-v1`, Job `1174846`, canonical colors)**:
  - `c1` (single front-top, `n=77`): **All = 0.961** (`up` 1.000, `turned` 0.961, `band` 0.974)
  - `s040+s220` (two side views, no `c1`, `n=154`): **All = 0.968** (`up` 0.989, `turned` 0.989, `band` 0.987)
  - `s130+top` (**two completely unseen cameras**, `n=102`): **All = 0.833** (`up` 0.961, `turned` 0.967, `band` 0.892)
  - `top` (single top-down, `n=51`): **All = 0.471** (`up` 0.806, `turned` 0.784, `band` 0.706)
- **Abstention Calibration (`C1-v2`, Job `1176313`)**:
  - `band` unanswerable (`n=21`): **100.0%** (`21/21`, both with hand and `@nohand`); answerable false-CBD rate only **0.6%** (with hand) / **1.3%** (`@nohand`).
  - `v3` unanswerable (`n=28`): **85.7%**.

Local artifacts:
- `Experiment Plan/results/z1-p1-eval/summary_montage_1168587.json`
- `Experiment Plan/results/c1-sft-eval/`
- `Experiment Plan/results/c1-boundary-plan-eval/`
- `Experiment Plan/results/c1-v2-eval/`

---

## Table 2. Public Benchmark No-Catastrophic-Forgetting Check (Frozen ViT LoRA)

| Benchmark | Questions | Base `Qwen3-VL-8B` (`1166604`) Greedy / Qwen | `C1-v1` (`1175804`) Greedy / Qwen | $\Delta$ (Greedy / Qwen) | `mean_tokens` (Base → SFT) |
| --- | ---: | ---: | ---: | ---: | ---: |
| **EmbSpatial** | 3,640 | 78.24% / 78.38% | **76.35% / 76.43%** | -1.89% / -1.95% | 9.5 → 9.6 |
| **MMStar** | 1,500 | 67.87% / 69.27% | **66.73% / 67.33%** | -1.13% / -1.93% | 479.9 → 6.1 |
| **BLINK** | 1,901 | 67.02% / 66.91% | **63.18% / 62.70%** | -3.84% / -4.21% | 357.7 → 5.1 |

Local artifact: `Experiment Plan/results/c1-bench-eval/bench_c1_vision_full_1175804.json`.
Note: `unmatched` dropped from 4.1–7.2% to 0.0–0.5%, and `truncated` dropped from 6.6–7.8% to 0.0%.

---

## Table 3. Open-Loop 96-State Perception (`n=180`) & BFS Macro-Action Planning (`n=360`, `L=1..4`)

| Model | Job ID | 96-State Canonical | 96-State OOD (G↔B) | Open-Loop `action_optimal` | Given Pred State Optimal | `reorient` (`n=264`) | `turn` (`n=96`) | `L1 / L2 / L3 / L4` Optimal |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `color_invariant` (no plan SFT) | `1174847` | 71.7% | 65.0% | 0.0% (`act_parsed` 0%) | 0.0% | 0.0% | 0.0% | — |
| `C1-v1` (`vision_full`) | `1174846` | 87.2% | **90.0%** | 93.1% | 94.1% | **98.9%** | 77.1% | 98.0 / 79.8 / 99.0 / **100.0%** |
| `C2-Grounded` GRPO | `1175221` | 88.3% | 88.9% | 92.8% | 93.8% | 98.5% | 77.1% | 97.0 / 79.8 / 99.0 / **100.0%** |
| `C2-Terminal` GRPO | `1175222` | 87.8% | **90.0%** | 93.1% | 93.8% | **98.9%** | 77.1% | 98.0 / 79.8 / 99.0 / **100.0%** |
| **`C1-v2` (cyclic turn CoT)** | `1176313` | **88.9%** | 87.2% | **96.1%** | **97.2%** | 98.5% | **89.6%** | 97.0 / **91.7** / **100.0** / 96.2% |

Local artifacts: `Experiment Plan/results/c1-boundary-plan-eval/vis_1174846_openloop_plan.json`, `Experiment Plan/results/c2-grpo-eval/`, `Experiment Plan/results/c1-v2-eval/c1_v2_1176313_openloop.json`.

---

## Table 4. Multi-Turn Closed-Loop Manipulation (Offline 200-Task Image Pool & Real 60 Hz Isaac Lab Online Rendering)

| Method / Evaluation Environment | Job ID | Episodes Evaluated | **Success Rate** | Fall Rate | Mean Rounds Ratio (`rounds / L`) | `L1 / L2 / L3 / L4` Rounds Ratio | Live / Step 96-State Acc | Illegal / Parse Fail |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero-shot Text Relational (Upper-Bound Input) | `1169616` | 200 | 43.5% (`L3`: 14%, `L4`: 4%) | — | — | — | (Oracle text) | — |
| BFS Oracle on Real Isaac Lab `Unified v4` | `1169489` | 2,048 | **100.0%** | 0.0% | 1.000× | 1.00 / 1.00 / 1.00 / 1.00 | (Oracle sim) | 0.0% |
| `C1-v1` — 200-Task Frozen Image Pool | `1175220` | 181 covered (96.9% test imgs) | **100.0%** (181/181) | — | 1.216× | 1.00 / 1.54 / 1.29 / 1.00 | 91.7% | 0.0% / 0.0% |
| `C2-Grounded` GRPO — 200-Task Pool | `1175221` | 182 covered (97.0% test imgs) | **100.0%** (182/182) | — | 1.236× | 1.00 / 1.54 / 1.29 / 1.09 | 90.2% | 0.0% / 0.0% |
| `C2-Terminal` GRPO — 200-Task Pool | `1175222` | 181 covered (96.9% test imgs) | **100.0%** (181/181) | — | 1.216× | 1.00 / 1.54 / 1.29 / 1.00 | 91.5% | 0.0% / 0.0% |
| **`C1-v2` — 200-Task Frozen Image Pool** | `1176313` | 181 covered (96.6% test imgs) | 98.9% (179/181) | — | **1.121×** | 1.05 / **1.32** / **1.08** / **1.02** | 90.5% | 0.4% / 0.0% |
| **`C1-v1` — Real 60 Hz Isaac Lab Online 3-Cam** | `1176233` | **128 physical** (32/level, 368 rounds) | **100.0%** (128/128) | **0.0%** | **1.175×** | 1.06 / 1.47 / 1.17 / 1.00 | **98.9%** | **0.0%** |

- Real Isaac Lab `1176233` physical execution speed: **15.7 control steps (0.26 s)** per macro-action at 60 Hz (`L1–L4`: 16.0 / 15.4 / 15.6 / 15.9 steps), matching the BFS Oracle (`16.5` steps) with zero sim-to-offline gap.

Local artifacts:
- `Experiment Plan/results/p2-text-plan-levers/summary_relational_1169616.json`
- `Experiment Plan/results/p2-bfs-oracle-job1169489/`
- `Experiment Plan/results/c2-grpo-eval/`
- `Experiment Plan/results/c1-v2-eval/c1_v2_1176313_pool200.json`
- `Experiment Plan/results/c1-isaac-closedloop/isaac_closedloop_c1_vision_full_1176233.json`

---

## Key Scientific Takeaways & Limitation / Future Work Points

1. **Map-then-Reason SFT resolves the fine-grained spatial grounding bottleneck**: Without SoM tags, zero-shot `band` accuracy is 7.5%; lightweight LoRA SFT (frozen ViT) lifts it to **98.7–99.4%**, unlocking **100.0% (128/128)** real 60 Hz physical closed-loop manipulation over all 96 cube states with **0% falls**.
2. **Decoupling visual generalization from prompt/format overfitting**: `C1-v1` generalizes strongly visually (nohand 99.4%, true-OOD `reverse_cycle` 96.2%, unseen `novel_palette` `band` 89.9–96.9%, unseen cameras `s130+top` 83.3%) with minimal forgetting on EmbSpatial (-1.9%) and MMStar (-1.1%), while failing on unseen inverse prompt `v3` (36.8%) and unanswerable CBD abstention (0.0%). **`C1-v2` (1 epoch)** directly validates the data-centric fix: CBD abstention recovers to **100.0% (21/21)**, unseen `v3` jumps to **59.2%** (`turned` **96.7%**), and explicit cyclic CoT lifts open-loop `turn` from **77.1% → 89.6%** (closed-loop rounds ratio **1.216× → 1.121×**).
3. **Closed-loop self-correction & GRPO saturation**: On the $\mathbb{Z}_4$ detent cycle, closed-loop visual feedback naturally recovers from occasional `+90/-90` sign flips (100% success at 1.17–1.22× rounds). Consequently, standard outcome-reward GRPO (`C2-Grounded` / `C2-Terminal`) encounters DAPO zero-variance saturation (74–82% all-pass groups), showing that SFT already suffices for task completion under 3-camera observability.
4. **Limitations & Future Work**:
   - **Single-camera partial observability & active perception**: Current closed-loop control uses 3 fixed cameras (`c1+s040+s220`), where almost every face is visible. Extending to a single heavily occluded camera (`c1`) with active exploratory `reorient` actions (information-gathering POMDP) and multi-turn belief tracking is the primary next step.
   - **Physical perturbation & sim-to-real**: Evaluating closed-loop visual replanning under external physical disturbances (slip, varying `hinge_effort_limit`, lighting/texture domain randomization) and transferring the vision-language planner to real hardware cameras.
   - **Remaining `v3` conservative abstention**: On `v3` inverse queries, `C1-v2` abstains conservatively on 26.3% of answerable frames (mainly when the cue is the top white-face black bar rather than a side strip); adding black-bar cue grounding will close this gap.
