# Stage 2 M1 plan: goal hidden, full k-sweep

User decision 2026-10-30: proceed to Stage 2 M1. The extra-seed work for Stage 1 and M2 is future work, tracked as GitHub issue #1. The scientific design follows v3 plan §M1. This file records the implementation choices and the order of steps. Tick each step off with its job ID.

## Design (from the v3 plan)

- **Task:** Stage 1 `Isaac-Repose-Cube-Shadow-Direct-v0`. Reward, physics, actions and PPO budget (5000 iterations, 2048 envs) are unchanged.
- **Actor:** sees the 157-D observation without the goal fields, i.e. 149-D, plus the intent.
  - Removed: `goal_rot` obs[64:68] and `quat_mul(object_rot, conj(goal_rot))` obs[68:72].
  - Kept: `in_hand_pos` obs[61:64]. It is the fixed in-hand target position (`default_root_state − 0.04 z`) and carries no goal-orientation information.
- **Critic:** asymmetric. RL-Games `central_value_config` sees the full clipped 157-D observation (goal included) and no intent, because the intent is a function of the state.
- **Intents:** computed from the full 157-D observation of the student's state, as in Stage 1. The obj intents need the goal; the expert intents run the Stage 1 expert on the full observation.
- **Expert and bases:**
  - expert: `expert-seed0` (Stage 1);
  - action PCA: the Stage 1 `INTENT_PCA_DIR`, as in array 1024454. Its `pca_basis.npz` holds both the 20-D and the 18-D finger bases.
- **Conditions:** 9 conditions × seeds 0, 1, 2 = 27 runs.
  - `blind` (Blind-B, lower bound);
  - `hand-2`, `hand-4`, `hand-8`, `hand-20`;
  - `wrist-2`, `wrist2-finger-2`;
  - `obj-1`, `obj-4` (Δθ = 0.5, same as Stage 1).
- **Upper bound (Goal-oracle):** the Stage 1 expert, P_E = 26.11 (no retrain, per plan).
- **Metric:** Recovery(c) = (P(c) − P(Blind-B)) / (P(Goal-oracle) − P(Blind-B)), plus N80 and AUC as in Stage 1.

## Implementation choices (not specified in the plan)

1. **Remove, not zero, the 8 goal dims.** For an MLP the two are equivalent. Removing makes the information loss explicit in the observation shape.
2. **`central_value_config`:** copied from Isaac Lab's `shadow_hand/agents/rl_games_ppo_ff_cfg.yaml`.
   - From that file: network [512, 512, 256, 128] elu, lr 5e-4 adaptive, kl 0.016, clip_value, normalize_input, truncate_grads.
   - Changed to match the Stage 1 actor: minibatch 32768, mini_epochs 5. The FF file uses 4 because its actor uses 4.
   - The actor's PPO settings stay exactly as in Stage 1.
3. **Gate before the grid** (v3 risk table: "Goal-oracle cannot reproduce Stage 1 → fall back to symmetric critic").
   - Train one goal-visible asymmetric-critic run, `blind`, seed 0, 5000 iterations.
   - Pass: final normalized successes ≥ 0.9 × 26.11. Stage 1 blind seeds reach 0.93–1.0.
   - If it fails, ask the user before falling back to a symmetric critic.
4. **Metadata:** `intent.json` gains `goal_hidden` and `asymmetric_critic`. `evaluate_students.py` reads `goal_hidden` from it, so the actor gets the same masked observation at evaluation.

## Steps

- [x] 1. Code (2026-10-30, local; not yet on the cluster):
  - wrapper flag `goal_hidden` (actor mask, `states` = full obs, `state_space` = 157);
  - `train_student.py`: `--goal_hidden` and `--asymmetric_critic`, which inject `central_value_config`;
  - `evaluate_students.py`: reads the flag;
  - `16`, `15` and `17` sbatch: `STUDENT_STAGE=m1` → Stage 1 task with goal hidden;
  - local CPU tests of the mask.
  - Done: `intent_modules.hide_goal` / `GOAL_FIELDS = [64:72]`; `intent_env_wrapper.py` is now the single wrapper (the duplicate in `train_student.py` is removed) with `goal_hidden` / `asymmetric_critic` and an overridden `state_space`; `train_student.py` writes `goal_hidden`, `asymmetric_critic`, `student_obs_dim`, `critic_states_dim` to `intent.json` and dumps `central_value_config` into `agent.yaml`; `evaluate_students.py` reads `goal_hidden`. Stages: `STUDENT_STAGE=m1` (both flags), `m1-gate` (asymmetric only), `stage1`/`m2` unchanged, unknown stage → exit 2. `17` needs no change (default task is Stage 1).
  - CPU test (stubbed Isaac Lab base wrapper, 18/18 PASS): mask removes exactly 64..71 and keeps order and `in_hand_pos`; M1 obj-4 actor = 149 + 4 with the intent from the full obs; `states` = full clipped 157-D; blind actor input invariant to the goal fields while `states` change; gate 157/157; default flags unchanged (no `states`). The real base class is only exercised by the step-2 GPU dry run.
  - Note: RL-Games' default `use_experimental_cv=True` keeps the actor's own value-head loss on (the shared trunk still gets a value gradient, but GAE uses the central value). Isaac Lab's FF config runs with the same default; left unchanged.
- [x] 2. GPU dry run (`15_student_dry_run.sbatch`, 256 envs × 3 iterations) for all 9 M1 conditions plus the goal-visible asymmetric gate. Check the actor obs dim (149 + k), states dim 157, and that evaluation of a dry-run checkpoint works.
  - Submitted 2026-10-30 (code = commit `9b17103`, synced; cluster checksums match; old remote copies in `~/cluster_smoke/.backup-pre-9b17103/`): `1040672` (`STUDENT_STAGE=m1`, 9 intents), `1040673` (`m1-gate`, blind), then `1040677` (`17`, `--array=0-1%1`, `afterok` on both) evaluating `runs/student-dry-run-job1040672/dry-obj-4-seed0` and `runs/student-dry-run-job1040673/dry-blind-seed0`. Log headers checked: Stage 1 task, expert `b9b7…`, correct flags.
  - **Result: all PASS.** `1040672` COMPLETED 0:0 (6:06), `student_dry_run_job_status=PASS`, 9/9 intents, actor obs blind 149, hand-2/wrist-2 151, hand-4/obj-4/wrist2-finger-2 153, hand-8 157, hand-20 169, obj-1 150; all 9 `critic_states_shape=(256, 157)`. `1040673` COMPLETED 0:0, gate blind actor 157, states 157. RL-Games log: "Adding Central Value Network", actor MLP input 153 and CV MLP input 157 (obj-4); checkpoint contains `assymetric_vf_nets`; `agent.yaml` holds `central_value_config` (mini_epochs 5, minibatch follows the actor's). `intent.json`: obj-4 `goal_hidden=true, asymmetric_critic=true, student_obs_dim=153, critic_states_dim=157`; gate `false/true/157/157`. `1040677_0/1` COMPLETED/PASS (~26 s per checkpoint); `evaluation.json` records `goal_hidden` True/153 (obj-4) and False/157 (gate).
- [x] 3. Gate run: goal-visible asymmetric `blind` seed 0, 5000 iterations, plus its evaluation.
  - Submitted 2026-11-10 (cluster code == `9b17103`, checksums checked): training `1103415` (`16`, `--array=0-0`, `STUDENT_STAGE=m1-gate`, blind, seed 0, 5000 it, 2048 envs; running on xgpf2; output `runs/student-m1-gate-blind-seed0-job1103415_0/`), full-curve evaluation `1103416` (`17`, `afterok`, 25 checkpoints, 1024 envs, seed 20261028; output `runs/eval-student-m1-gate-blind-seed0-job1103416_0/`). Pass if final (ep_5000) successes/episode ≥ 23.50 (0.9 × 26.11).
  - **Result: PASS.** `1103415` COMPLETED 0:0 (1:47:34, `student_job_status=PASS`), `1103416` COMPLETED/PASS (25 checkpoints). Final ep_5000 = **25.755** successes/episode (0.986 of P_E), reward 7134, fall 6.3%. N80 2375 iterations, AUC 0.649; Stage 1 blind seeds 0–2 (symmetric critic): final 0.947–1.0, N80 1546–3189, AUC 0.598–0.800, so the asymmetric critic is inside the blind seed spread. Training is ~15% slower (1:47 vs ~1:34). Summary: `Experiment Plan/results/m1-gate-job1103416/`.
- [ ] 4. Grid: array of 27 runs, `--array=0-26%4`.
  - Submitted 2026-11-10: training `1104420` (`16`, `--array=0-26%4`, `STUDENT_STAGE=m1`, intents `blind hand-2 hand-4 hand-8 hand-20 wrist-2 wrist2-finger-2 obj-1 obj-4` × seeds `0 1 2`, task i → intent i//3, seed i%3; Stage 1 PCA `job1024096`; 5000 it, 2048 envs, Δθ 0.5; cluster code == `1235046`). Outputs `runs/student-m1-<intent>-seed<s>-job1104420_<i>/`. Task 0 header checked (m1, both flags). Submitted only.
  - Cluster limit: **32 queued jobs per user** (each array task counts; `AssocMaxSubmitJobLimit`). So evaluation goes in batches with `aftercorr:1104420` and the same `EVAL_RUN_DIRS` list (saved on the cluster as `~/cluster_smoke/m1_grid_eval_dirs_1104420.txt`, 27 entries, index i ↔ training task i): batch 1 = `1104422` (`--array=0-4%4`). Submit the remaining indices 5–26 as queue room frees up (keep queued ≤ 32 and running GPUs ≤ 4).
  - 2026-11-10 07:28: tasks 0–3 trained and evaluated (all PASS). Batch 1 evaluations overlapped 4 running training tasks (up to ~6 GPUs, over the 4-GPU cap). Fix: training throttle lowered to %3 (`scontrol update JobId=1104420 ArrayTaskThrottle=3`), evaluations run one at a time (1104422 throttle 1). Batch 2 = `1104730` (`--array=5-12%1 --dependency=aftercorr:1104420,afterany:1104422`). Next batches: same pattern with `afterany:<previous eval batch>`.
- [ ] 5. Evaluation: array of 27 (`17_evaluate_students.sbatch`), the same evaluation settings as Stage 1.
- [ ] 6. Analysis: N80, AUC and Recovery per condition; tests vs Blind-B; commit the results and update `CONTEXT.md`.
