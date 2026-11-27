# What Should a Slow Planner Tell a Fast Dexterous Hand?

**Content, rate, and validity windows in low-dimensional intent interfaces** — CS5478 (NUS SoC), Team 34.

https://github.com/user-attachments/assets/4ac2db1f-b76f-4386-b123-20ce4ee6de7c

Vision-language models plan at 0.2–2 Hz; a dexterous hand needs 60 Hz closed-loop control.
This project asks what a low-dimensional *intent* sent from a slow planner to a fast Shadow Hand controller should contain.
We compare three abstraction levels in Isaac Lab:

- action synergies (PCA of expert actions)
- object waypoints
- goal-level commands

The three tasks are:

- 3D cube reorientation
- z-axis face turn
- a simplified Rubik's-like cube that combines whole-cube reorientation with ±90° top-layer twists

## Key findings

- At 60 Hz, 8-D action synergies match goal commands (about 87% expert recovery).
- At ≤ 10 Hz, or with ≥ 0.1 s latency, the ranking reverses. Staleness between updates collapses the action and waypoint interfaces. Only goal commands (≥ 8 bits per component) degrade predictably, following a waiting-time law `P(d) ≈ 600 / (T0 + d)`.
- We trained a unified 60 Hz multi-skill controller, **Unified v4**, that holds the cube through 4 s planner pauses without dropping it.
- **Unified v4** is paired with **Qwen3-VL-8B-Instruct**, fine-tuned with a Map-then-Reason LoRA (0.51% trainable parameters, frozen ViT). It reaches **100% task completion over 128 real-time closed-loop episodes** (L = 1–4 moves) with zero drops.

## Demos

Full demo (Acts 0–5):

https://github.com/user-attachments/assets/bb6c1e9f-7291-45bb-9175-90c29993c85e

| Clip | What it shows |
| --- | --- |
| VLM closed loop (`vlm_end_to_end_ep0`) | The VLM observes, perceives and plans; Unified v4 executes the plan at 60 Hz |
| Dark low light (`vlm_gen3_dark_lowlight`) | Zero-shot lighting shift; 32/32 episodes solved in the benchmark |
| Swapped colours (`vlm_gen4_ood_colour_rev`) | Out-of-distribution cube colours; 3/3 optimal rounds |
| Unseen prompt (`vlm_gen2_unseen_prompt`) | Instruction wording not seen in training |

<!-- VIDEO: optional, add the clips above here the same way -->

Videos are not stored in the repository. `Docs/Demo/make_github_demo.py` and `Docs/Demo/make_vlm_closedloop_video.py` render them from the evaluation recordings.

## Repository layout

| Path | Contents |
| --- | --- |
| `Experiment Plan/cluster_smoke/` | Training, evaluation, recording and VLM pipeline code; Slurm scripts (`NN_*.sbatch`) |
| `Experiment Plan/*.md` | Experiment plans (rate sweep, face turn, goal-level RL, scheme comparison) |
| `Experiment Plan/results/` | Evaluation summaries and analyses, one folder per Slurm job |
| `Docs/Milestone/data/` | Per-checkpoint evaluation summaries used for figures |
| `Docs/Final presentation/` | Final deck, scripts and figure generation |
| `report/` | Final report (LaTeX, NeurIPS 2024 style); see `report/README.md` |
| `CONTEXT.md`, `HANDOFF.md` | Current status and the full experiment history with job IDs |

## Setup

- Isaac Lab v2.3.1 and Isaac Sim 5.1.0, with Python 3.11, PyTorch 2.7 and RL-Games 1.6.1.
- Base task: `Isaac-Repose-Cube-Shadow-Direct-v0`.
- Experiments ran on the NUS SoC Slurm cluster (T4 for simulation, H100 for VLM training and serving).
- Checkpoints, datasets and raw rollouts are kept on the cluster, not in Git.
