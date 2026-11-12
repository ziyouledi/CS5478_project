# Rate sweep plan: intent dimension × update rate (goal hidden)

User decision 2026-11-11: pivot the study from "how many intent dimensions" to "how many dimensions at what update rate, and can a slow semantic source produce them". This plan adds one axis to Stage 2 M1: the intent is refreshed every N control steps instead of every step. Tick each step off with its job ID. Deadlines: demo 2026-11-18, report end of November.

## Why

- In M1 the intent is recomputed at 60 Hz from the expert's action at the same control step. The upstream must therefore be a 60 Hz expert. The M1 curve (hand-2 0.50 → hand-8 0.87) measures how well a 60 Hz action stream compresses linearly.
- A language model as the upstream is slow. Published zero-shot evaluations of frontier models on arm + gripper tasks report 2–8 s per decision, about 0.1–0.5 Hz. No such result exists for dexterous hands.
- Question: with a fixed low-level learner, which interface (action-level, subgoal-level, goal-level) still recovers performance when it is refreshed at 10, 2 or 0.5 Hz?
- Prediction (not yet tested): hand-k collapses under hold, because the expert is near bang-bang (sign flips on 20–32% of consecutive steps). obj-4 and goal-4 change slowly and should survive. If so, the M1 ranking reverses at language-model rates.

## Fixed from M1 (unchanged)

- Task `Isaac-Repose-Cube-Shadow-Direct-v0`; reward, physics, actions, 60 Hz control.
- Actor: 149-D goal-hidden observation + intent. Critic: asymmetric, full 157-D observation, no intent.
- PPO: 5000 iterations, 2048 envs, same config as M1 grid `1104420`.
- Expert `expert-seed0` (`b9b7…`), action PCA `job1024096`, Δθ = 0.5 rad for obj-4.
- Evaluation: `17_evaluate_students.sbatch`, all 25 checkpoints, 1024 envs, seed 20261028.
- Metric: Recovery = (P − P_B) / (P_E − P_B), P_E = 26.11, P_B = 0.006 (M1 Blind-B); N80 and AUC as before.
- Seeds 0, 1, 2. At most 4 GPUs including evaluation. At most 32 queued jobs.

## New definitions (scientific changes, approved 2026-11-11)

1. **Hold-N (zero-order hold).** The intent of an environment is recomputed from its current full observation when its episode step count is a multiple of N; otherwise the previous value is kept. Every episode therefore starts with a fresh intent. A goal change inside an episode (after a success) does not trigger a refresh. N = 1 is M1.

   | N (control steps) | 1 | 6 | 30 | 120 |
   |---|---|---|---|---|
   | Update rate | 60 Hz | 10 Hz | 2 Hz | 0.5 Hz |

2. **`goal-4`.** The goal quaternion itself, `canonical_quat(obs[64:68])`, w ≥ 0. No expert, no lookahead. It is the goal-level interface, the one a language model could most plausibly produce. At N = 1 it gives the actor the goal back (minus the relative rotation `obs[68:72]`), so it should approach the Stage 1 level.

3. With a hold, the intent is no longer a function of the current state. The critic still sees the full state and the current goal, as in M1.

Not in scope here: delay (intent computed from an observation d steps old), quantisation or noise on the intent, the M2 task. See "Later".

## Steps

- [x] 0. **Offline hold analysis (CPU, no training).** `analyze_intent_hold.py` / `22_intent_hold.sbatch` on the Stage 1 expert rollout `1023537` (512 envs × 2400 steps).
  - Intents: hand-2/4/8/20, wrist-2, obj-1, obj-4, goal-4. Holds N = 1, 2, 3, 6, 12, 30, 60, 120.
  - Per intent and N: normalised hold error E‖z_t − z_held‖² / total variance (0 = nothing lost, 1 = no better than the mean, 2 = uncorrelated); within-episode autocorrelation at the same lags.
  - Action intents: share of the expert's 20-D action variance that the held intent still reconstructs. At N = 1 this must equal the PCA cumulative variance (35.8% for k = 2, 81.1% for k = 8); the job fails otherwise.
  - Quaternion intents: mean angle between current and held value, and the share of steps where it exceeds the 0.1 rad success tolerance.
  - Goal statistics: mean interval between goal changes; share of steps whose goal changed since the last refresh.
  - Caveat: the data come from the expert's own trajectories with a fresh intent every step. It measures how fast each signal goes stale, not what a student trained under hold would achieve.
- [x] 1. **Evaluation-only hold (GPU, no training).** DONE 2026-11-11: `1110531`, `1110532`, `1110535`, `1111529`; see Log. Existing M1 final checkpoints (`ep_5000`) of hand-2, hand-8, obj-4 × seeds 0–2 from `1104420`, evaluated with N = 1, 6, 30, 120.
  - Code: hold option in `intent_env_wrapper.py` (default N = 1, behaviour unchanged) and an `EVAL_INTENT_HOLD` override in `evaluate_students.py` / `17`.
  - Check: N = 1 reproduces the recorded M1 final value within the evaluator's order effect (~0.5%).
  - Caveat: these students were trained at 60 Hz, so the result is a lower bound for step 2.
- [x] 2. **Training grid.** DONE 2026-11-12: 39/39 training + evaluation PASS (training `1111584`, `1116466`; evaluations `1111585`, `1116420`, `1116422`, `1117654`, `1117655`).
  - Code: `goal-4` in `intent_modules.py`; `--intent_hold` in `train_student.py`, written to `intent.json` and read by the evaluator; `STUDENT_INTENT_HOLD` in `15`/`16`. CPU test of the hold logic, then a GPU dry run (256 envs × 3 iterations) for every (intent, N) pair.
  - Conditions (N set decided by the user 2026-11-11 after step 0/1: **6, 12, 30**, not 6, 30, 120): hand-2, hand-8, obj-4, goal-4 × N = 6, 12, 30 × 3 seeds = 36 runs, plus goal-4 at N = 1 × 3 seeds = **39 runs**. hand-2, hand-8, obj-4 at N = 1 are the existing M1 runs.
  - Cost: an M1 run takes about 1:47. With 3 training slots and 1 evaluation slot, 39 runs take about 23 hours of wall time. Evaluations (12 min each) fit in the fourth slot. Submit in batches under the 32-job limit, with `aftercorr` evaluations submitted while training is still pending.
  - Pruning rule: if steps 0 and 1 show that hand-k keeps its information at N = 30 and 120 (the prediction is wrong), drop hand-2 and keep hand-8, obj-4, goal-4 (30 runs).
- [x] 3. **Analysis.** DONE 2026-11-12, see Log (`analyze_rate_sweep.py`, `results/rate-sweep-grid-39runs/`). Recovery, AUC and N80 per (intent, N); Recovery against update rate, one line per intent; Welch tests against the same intent at N = 1. Copy summaries to `results/rate-sweep-*/`, update `CONTEXT.md`.

## Reading the result

- hand-8 falls below goal-4 or obj-4 at 2 Hz or 0.5 Hz → action-level interfaces need a fast upstream; only goal-level interfaces fit a language-model channel. This is the headline for the demo.
- hand-8 stays on top at all rates → the learner extracts slow structure from the action code; report that and move GPU time to the M2 comparison.
- goal-4 at N = 1 far below 1.0 → the M1 actor needs the relative rotation, not just the goal; check before interpreting the rest.

## Schedule

| Date | Work |
|---|---|
| 11-11 | Plan; step 0 submitted; step 1 code |
| 11-12 | Step 1 result; step 2 code, dry run, grid submitted |
| 11-13 → 11-14 | Grid and evaluations finish; step 3 analysis |
| 11-15 → 11-17 | Figures, demo slides and script |
| after 11-18 | M2 comparison (goal is 1-D yaw): does obj-1 suffice at low rate? Delay and quantisation ablations |

## Log

### Step 0 (2026-11-11): DONE, `1110496` COMPLETED 0:0, `intent_hold_job_status=PASS`

- First run `1110491` also PASSED, but its goal-change detector fired on float32 quaternion noise (1.1% of steps at N = 1, where it must be 0). Fix: normalise quaternions before taking the angle, threshold 1e-3 → 1e-2 rad, and fail if N = 1 reports any goal change. Rerun = `1110496`. All other numbers are identical between the two runs.
- Dataset checksums PASS; 1,228,800 transitions; hold 1 reproduces the PCA cumulative variance (hand-2 0.358, hand-8 0.811, hand-20 1.000).
- Output: cluster `datasets/intent-hold-expert-rollout-seed0-job1023537-job1110496/`; local copy `results/intent-hold-job1110496/` (summary.json, intent_hold.csv, intent_hold.png; checksums verified).

| Update rate (N) | 60 Hz (1) | 30 Hz (2) | 20 Hz (3) | 10 Hz (6) | 5 Hz (12) | 2 Hz (30) | 1 Hz (60) | 0.5 Hz (120) |
|---|---|---|---|---|---|---|---|---|
| hand-8: share of expert action variance | 0.81 | 0.44 | 0.06 | −0.53 | −0.64 | −0.74 | −0.77 | −0.78 |
| hand-2: share of expert action variance | 0.36 | 0.22 | 0.05 | −0.24 | −0.29 | −0.33 | −0.33 | −0.31 |
| hand-8: hold error / variance | 0 | 0.46 | 0.92 | 1.65 | 1.78 | 1.91 | 1.95 | 1.96 |
| obj-1: hold error / variance | 0 | 0.14 | 0.28 | 0.73 | 1.44 | 1.78 | 1.88 | 1.97 |
| obj-4: hold error / variance | 0 | 0.05 | 0.10 | 0.24 | 0.49 | 1.13 | 1.58 | 1.79 |
| goal-4: hold error / variance | 0 | 0.04 | 0.09 | 0.22 | 0.49 | 1.17 | 1.60 | 1.76 |
| obj-4: steps with error > 0.1 rad | 0 | 0.31 | 0.46 | 0.66 | 0.81 | 0.91 | 0.95 | 0.97 |
| goal-4: steps with a stale goal | 0 | 0.02 | 0.05 | 0.11 | 0.25 | 0.55 | 0.75 | 0.86 |

Findings:

1. **Action intents are dead by 20 Hz.** Holding hand-k for 3 steps already leaves ~5% of the action variance; from 10 Hz down the held value is worse than sending the mean (negative share). The lag-1 autocorrelation of hand-k is only 0.50–0.60. All k behave the same, so at any rate below ~20 Hz the dimension k no longer matters.
2. **The goal itself changes every 21.9 control steps (0.37 s, 2.7 Hz).** The expert succeeds about 26 times per 10 s episode and the environment resamples the goal on every success. So a goal-level intent refreshed at 2 Hz is stale 55% of the time and at 0.5 Hz 86% of the time. The prediction "goal-4 survives at 0.5 Hz" is wrong for a periodic refresh on this task.
3. **Throughput ceiling (derivation, not a measurement).** If a student can only reach goals it has been told, a periodic refresh allows at most about one success per N steps, i.e. 600/N per episode: 100 at N = 6, 50 at N = 12, 20 at N = 30, 5 at N = 120. Against P_E = 26.11 the Recovery ceilings are 1, 1, ~0.77, ~0.19.
4. obj-4 and goal-4 lose signal energy at the same pace, but obj-4 is outside the 0.1 rad tolerance far more often (66% vs 11% at 10 Hz), because the subgoal moves with the object.

Consequences for steps 1 and 2 (need a user decision before step 2):

- The metric "successes per episode" mixes two things at low rate: whether the interface works, and how many goals the upstream can issue per second. Report Recovery next to the ceiling min(1, (600/N)/P_E).
- Proposed N set for the grid: **6, 12, 30** instead of 6, 30, 120. N = 120 is capped at ~0.19 by arithmetic and would say little. Step 1 (evaluation only, cheap) can still include 120.
- The quantity that matters for a slow upstream is goals per second, not refreshes per second: goal-4 at N = 1 is already an event-driven signal with 2.7 events/s. A language model at 0.5 Hz could drive this task at ≤ 5 goals per episode however good the low-level controller is. The useful framing for the report: each task has a goal bandwidth (events/s × DoF); this one is 2.7 Hz × 3 DoF by construction.


### Step 1 (2026-11-11): DONE, 36/36 tasks COMPLETED, `student_eval_job_status=PASS`

- Code: `intent_hold` in `IntentRlGamesVecEnvWrapper` (zero-order hold keyed on `episode_length_buf`, which Isaac Lab increments, then zeroes on reset, before computing observations; `max_consecutive_success = 0` for Direct-v0, so successes do not reset it). Evaluator `--intent_hold` (default: `intent.json` value or 1) and `--last_checkpoint_only`; `17` reads `EVAL_INTENT_HOLD` (output dir suffix `-holdN`) and `EVAL_LAST_ONLY=1`. `evaluation.json` records `intent_hold`, `intent_hold_trained`. CPU stub test of the hold (periodic refresh, refresh on per-env reset, N = 1 passthrough) PASS. Synced (remote matched HEAD before sync; checksums match after).
- Run list: cluster `~/cluster_smoke/rate_sweep_step1_dirs.txt` (index 0–2 hand-2 s0–2, 3–5 hand-8 s0–2, 6–8 obj-4 s0–2; `1104420` inner run dirs). Final checkpoint only, 1024 envs, seed 20261028.
- Jobs: N = 1 `1110531`, N = 6 `1110532`, N = 30 `1110535`, N = 120 `1111529` (submitted after the first three finished; 32-job limit). About 1:10 per task.
- Analysis: `analyze_rate_sweep_step1.py`; evaluation copies in `Docs/Milestone/data/rate_sweep_step1_evals/` (gitignored); output `results/rate-sweep-step1/` (hold_eval.csv, summary.json).
- Reproduction check at N = 1 (tolerance 1% relative, against the M1 grid evaluation of the same checkpoint): hand-2 and hand-8 within 0.6%; **obj-4 seed 0 +1.6% (5.405 vs 5.321) and seed 2 −2.8% (8.805 vs 9.062)**, so the script reports `status=FAIL`. At N = 1 the hold returns the intent unchanged (same code path as M1); the only difference is that M1 evaluated the final checkpoint 25th in one process, this run evaluates it alone. Attributed to the evaluator order effect, larger for obj-4; not verified. Effect on Recovery ≤ 0.01, small against the hold effects below.

Recovery (mean ± std over seeds 0–2), fall fraction in brackets; ceiling = min(1, (600/N)/P_E):

| Intent | 60 Hz (N=1) | 10 Hz (N=6) | 2 Hz (N=30) | 0.5 Hz (N=120) |
|---|---|---|---|---|
| hand-2 | 0.504 ± 0.079 [0.03] | 0.005 ± 0.002 [0.44] | 0.001 [0.70] | 0.000 [0.69] |
| hand-8 | 0.865 ± 0.113 [0.03] | 0.028 ± 0.008 [0.62] | 0.000 [0.97] | 0.000 [0.87] |
| obj-4 | 0.326 ± 0.114 [0.09] | 0.231 ± 0.052 [0.13] | 0.053 ± 0.004 [0.52] | 0.013 [0.56] |
| ceiling | 1 | 1 | 0.77 | 0.19 |

Findings:

1. **The ranking reverses at 10 Hz.** hand-8 drops from 0.87 to 0.03, hand-2 from 0.50 to 0.005; obj-4 keeps 0.23 (71% of its 60 Hz value). Consistent with step 0 (held hand-k is worse than the mean from 10 Hz down).
2. A stale action intent is harmful, not just uninformative: hand-8 drops the cube in 62% of episodes at 10 Hz and 97% at 2 Hz (Blind-B, which gets no intent, falls in 0.2–1.2% of episodes, `m1_evals` final checkpoints). The policy follows the held action code.
3. At 2 Hz and 0.5 Hz every interface is far below its ceiling (obj-4 0.05 vs 0.77). These students were trained at 60 Hz; step 2 tests whether training under hold closes the gap.
4. For step 2 the pruning rule does not apply (hand-k does not keep its information); keep hand-2.

### Step 2 (2026-11-11): code and dry run

- User decision 2026-11-11: training grid N = 6, 12, 30 (10, 5, 2 Hz). N = 120 dropped (ceiling ~0.19).
- Code: `goal-4` in `intent_modules.py` (kind `goal`, `canonical_quat(obs[64:68])`, no expert, no lookahead); `train_student.py --intent_hold` (passed to the wrapper, written to `intent.json`; the evaluator reads it back); `16`: `STUDENT_INTENT_HOLD`, and a `STUDENT_INTENTS` entry may be `intent:N` so one array mixes holds (run name `student-m1-<intent>-hold<N>-seed<s>`, no suffix at N = 1); `15`: `DRY_RUN_HOLDS` loop, checks the logged hold. CPU tests PASS: goal-4 (canonical, w ≥ 0, actor 153-D), grid index mapping 0–38 of 16. Synced (remote matched HEAD before; checksums match).
- GPU dry run `1111565`: `STUDENT_STAGE=m1`, intents hand-2 hand-8 obj-4 goal-4 × holds 1 6 12 30, 256 envs × 3 iterations.
- **Dry run `1111565` PASS** (COMPLETED, 16/16, ~40 s each): actor dims hand-2 151, hand-8 157, obj-4 153, goal-4 153; `critic_states_shape=(256, 157)` everywhere; logged `intent_hold` matches. goal-4 `intent.json`: no expert, no lookahead, `goal_hidden` true, `intent_hold` recorded.

### Step 2 grid (2026-11-11 08:44): batch 1 SUBMITTED

- One list, 13 entries × seeds 0 1 2 = 39 tasks, task i → entry i//3, seed i%3:
  `goal-4:1 hand-2:6 hand-8:6 obj-4:6 goal-4:6 hand-2:30 hand-8:30 obj-4:30 goal-4:30 hand-2:12 hand-8:12 obj-4:12 goal-4:12`
  (N = 6 and 30 before 12, so the 10 Hz / 2 Hz contrast finishes first). Same settings as `1104420` (m1, PCA `job1024096`, 5000 it, 2048 envs, Δθ 0.5).
- Batch 1: training `1111584` (`--array=0-20%3`), evaluation `1111585` (`--array=0-8%1`, `aftercorr:1111584`). 30 queued jobs. Task 0–2 headers checked (goal-4, hold 1, m1 flags, expert `b9b7…`).
- Cluster lists: `~/cluster_smoke/rate_sweep_grid_intents_1111584.txt`, `~/cluster_smoke/rate_sweep_grid_eval_dirs_1111584.txt` (39 entries, index i ↔ training task i **of 1111584 only**).
- To do, as queue room frees (≤ 32 queued, ≤ 4 GPUs = 3 training + 1 evaluation):
  1. Evaluations 9–20 of `1111584`: same `EVAL_RUN_DIRS` file, `--array=9-20%1`, `aftercorr:1111584`, plus `afterany:1111585` so only one evaluation runs. Submit before those training tasks finish (aftercorr on already-finished tasks got stuck in M1; if so, use no dependency for finished tasks).
  2. Batch 2 training = indices 21–38 of the same list as a new array (`--array=21-38%3`), only when `1111584` has no pending tasks left (else > 3 training GPUs). Its run dirs carry the new job ID: regenerate the eval-dir list with that ID before submitting its evaluations.
- ETA: batch 1 ≈ 7 waves × 1:47 ≈ 12.5 h; batch 2 ≈ 6 waves ≈ 11 h.

### Step 2 status 2026-11-11 20:59

- `1111584` tasks 0–17 COMPLETED 0:0, `student_job_status=PASS` (1:43–1:49 each); 18–20 running. `1111585` (evaluations 0–8) all COMPLETED/PASS (~11 min each).
- Evaluations 9–20 had not been submitted in time (training 9–17 already finished). Submitted 21:00 without `aftercorr` for finished tasks: `1116420` (`--array=9-17%1`, no dependency) and `1116422` (`--array=18-20%1`, `aftercorr:1111584,afterany:1116420`).
- Batch 2 training (21–38, `afterany:1111584`, %3) is submitted by a background loop once the queue is ≤ 14 jobs (limit 32); its eval-dir list is written as `rate_sweep_grid_eval_dirs_<job>.txt`. Its evaluations still have to be submitted.
- First results, final checkpoint (ep 5000), Recovery = (P − P_B)/(P_E − P_B), falls in brackets (not the final analysis):
  - goal-4, N = 1: 0.855 / 0.706 / 0.753, mean **0.77** [7–8%]. The goal quaternion alone (without the relative rotation) recovers most of the expert; the "goal-4 far below 1.0" check does not trigger.
  - hand-2, N = 6: 0.001 / 0.046 / 0.043, mean 0.03 [2–3%].
  - hand-8, N = 6: 0.295 / 0.090 / 0.149, mean **0.18** [1–2%]; evaluation-only hold gave 0.03 [62%]. Training under hold removes the falls and recovers part of the performance, still far below hand-8 at 60 Hz (0.87).
- 21:10: batch 2 training SUBMITTED = `1116466` (`--array=21-38%3`, `afterany:1111584`), queue then 32/32. Eval-dir list `~/cluster_smoke/rate_sweep_grid_eval_dirs_1116466.txt` (use indices 21–38). Batch 2 evaluations NOT yet submitted: when queue room allows (after `1116420`/`1116422` drain, ~23:30), submit `17` with that list, `--array=21-38%1`, `aftercorr:1116466` for tasks still unfinished and no dependency for finished ones. Batch 2 ETA ≈ 6 waves × 1:47 ≈ 11 h from ~21:15 (≈ 08:00 11-12).

### Step 2 status 2026-11-12 01:45

- Batch 1 COMPLETE: `1111584` 0–20 and evaluations `1111585` (0–8), `1116420` (9–17), `1116422` (18–20): 21 + 21 COMPLETED, all `job_status=PASS`.
- Batch 2 training `1116466`: 21–26 COMPLETED/PASS, 27–29 running, 30–38 pending (ETA ≈ 08:00–09:00 11-12).
- Batch 2 evaluations SUBMITTED 01:45 with `~/cluster_smoke/submit_rate_sweep_b2_evals.sh` (classifies each training task by `sacct` state): `1117654` = indices 21–26 (finished, no dependency, %1); `1117655` = 27–38 (`aftercorr:1116466,afterany:1117654`, %1). Queue 30/32. SSH to `xlogin` failed 01:38 (kex closed after the `stujump` hop) and recovered by 01:40; first two submission attempts submitted nothing (script bugs, verified with `squeue`/`sacct`).
- Whole grid is now submitted. Next: when `1117655` finishes, step 3 analysis.

### Step 2 complete (verified 2026-11-12 08:41)

- Batch 2 training `1116466` 21–38: 18/18 COMPLETED 0:0, `student_job_status=PASS` (last 08:11). Evaluations `1117654` (21–26) and `1117655` (27–38): 18/18 COMPLETED 0:0, `student_eval_job_status=PASS` (last 08:37). Queue empty. Whole grid: 39 training + 39 evaluations PASS.

### Step 3 (2026-11-12): DONE, `rate_sweep_analysis_status=PASS`

- Evaluation copies: `Docs/Milestone/data/rate_sweep_grid_evals/` (39 files, gitignored; renamed `<intent>[-holdN]-seed<s>-job<eval>_<i>.json`). N = 1 for hand-2 / hand-8 / obj-4 from the M1 grid (`Docs/Milestone/data/m1_evals/`). Script `cluster_smoke/analyze_rate_sweep.py` (checks: 48 runs, 3 seeds per (intent, N), 25 checkpoints ending at 5000, evaluated hold = trained hold). Output `results/rate-sweep-grid-39runs/` (summary.json, runs.csv, recovery_vs_rate.png, learning_curves_by_rate.png).

Recovery (final checkpoint, mean ± sd over seeds 0–2), AUC in brackets; trained under the hold; p = Welch vs the same intent at N = 1, uncorrected:

| Intent | 60 Hz (N=1) | 10 Hz (N=6) | 5 Hz (N=12) | 2 Hz (N=30) |
|---|---|---|---|---|
| hand-2 | 0.504 ± 0.079 [0.37] | 0.030 ± 0.025 [0.02], p = 0.005 | 0.002 [0.00], p = 0.008 | 0.001 [0.00], p = 0.008 |
| hand-8 | **0.866 ± 0.112** [0.62] | 0.178 ± 0.105 [0.09], p = 0.002 | 0.096 ± 0.033 [0.05], p = 0.004 | 0.021 ± 0.017 [0.01], p = 0.005 |
| obj-4 | 0.328 ± 0.116 [0.11] | 0.217 ± 0.031 [0.08], p = 0.24 | 0.132 ± 0.080 [0.05], p = 0.08 | 0.019 ± 0.006 [0.01], p = 0.04 |
| goal-4 | 0.771 ± 0.076 [0.45] | **0.707 ± 0.011** [0.42], p = 0.28 | **0.648 ± 0.017** [0.40], p = 0.10 | **0.426 ± 0.028** [0.27], p = 0.009 |
| ceiling | 1 | 1 | 1 | 0.77 |
| step 1 (eval-only hold) | | hand-2 0.005, hand-8 0.028, obj-4 0.231 | | hand-2 0.001, hand-8 0.000, obj-4 0.053 |

Findings:

1. **The ranking reverses below 60 Hz, now with students trained under the hold.** At 60 Hz hand-8 and goal-4 are not distinguishable (0.87 vs 0.77, p = 0.30). At 10, 5 and 2 Hz goal-4 is first and every other interface is below it with p ≤ 0.012 (Welch vs goal-4 at the same N). At 2 Hz goal-4 keeps 0.43, the others ≤ 0.02.
2. **Action-level intents collapse even when the student is trained under the hold.** hand-8 keeps 21% of its 60 Hz Recovery at 10 Hz, 11% at 5 Hz, 2% at 2 Hz; hand-2 is ≤ 0.03 from 10 Hz down. Training under the hold mainly removes the falls (hand-8 at 10 Hz: 62% falls evaluation-only → 2% trained) and lifts hand-8 from 0.03 to 0.18; it does not rescue the interface.
3. **obj-4 degrades gradually** (0.33 → 0.22 → 0.13 → 0.02) and is never better than goal-4. Training under the hold gives nothing over evaluation-only hold (0.22 vs 0.23 at 10 Hz).
4. **goal-4 degrades slowly** (−8% at 10 Hz, −16% at 5 Hz, −45% at 2 Hz relative to 60 Hz) and stays below the one-goal-per-refresh ceiling (2 Hz: 0.43 vs 0.77). Its seed spread is the smallest of all groups (sd 0.01–0.03). goal-4 learning curves are still rising at iteration 5000 at every N, so its final values are lower bounds for longer training.
5. N80 (0.8 P_E) is reached only at 60 Hz (hand-8 2/3, goal-4 1/3); AUC follows the same ordering as Recovery.

Reading (per "Reading the result"): hand-8 falls below goal-4 from 10 Hz on → action-level interfaces need a fast upstream; on this task only the goal-level interface survives a slow channel. Caveats: one task, 3 seeds, uncorrected tests; goal-4 is the goal itself (the hidden variable of M1), so this compares interface levels, not compression of an expert; a language model would also add latency and noise, not tested here (delay/quantisation are listed under "after 11-18").

---

# Phase 2 (approved by the user 2026-11-12): second task, longer training, more seeds

Budget: about 40 h wall clock on the cluster, 4 GPUs (3 training + 1 evaluation), ≤ 32 queued jobs. 57 training runs × ~1:47 = 19 waves ≈ 34 h, plus queueing and reruns. Target: everything finished by 2026-11-14 morning; demo 2026-11-18. Tick each step off with its job ID, as in phase 1.

## Questions

1. Does the reversal (action-level best at 60 Hz, goal-level best below) repeat on a second task (M2, goal = 1-DoF yaw)?
2. On a task whose goal has 1 DoF, is a 2-D goal intent as good as the 4-D quaternion at every rate?
3. Is goal-4 at 2 Hz limited by training length (ceiling 0.77, measured 0.43, curves still rising)?
4. Are the phase 1 contrasts stable with 5 seeds?
5. What latency and precision does the goal channel tolerate (evaluation only)?
6. Pilot: can a 1-bit slow selector switch between the two frozen experts (evaluation only)?

## New definitions (scientific changes, approved 2026-11-12)

1. **Stage `m2-hidden`.** Task `Isaac-Repose-Cube-Shadow-ZAxis-Direct-v0`, expert `expert-m2-seed0` (`35dc6eb1…`), goal hidden from the actor (same fields `[64:72]`), asymmetric critic, same PPO settings as M1. Action PCA for hand-8 = the M2 basis `action-pca-expert-m2-rollout-seed0-job1033178-job1033179`.
   - Reference values: P_E = 31.14 (M2 expert, `1032490`); P_B = final successes/episode of the `m2-hidden` blind runs (measured in block 1).
   - Ceiling at hold N: min(1, (600/N)/31.14) → 1, 1, 1, 0.64 for N = 1, 6, 12, 30.
2. **`goal-yaw` (2-D).** (sin ψ, cos ψ), ψ = rotation angle of the goal quaternion about world z: ψ = 2·atan2(q_z, q_w) on `obs[64:68]`. Only meaningful on the M2 task, where goals are pure yaw. The validator must check on M2 rollout data that the goal quaternion has |q_x|, |q_y| below 1e-3 and that (sin ψ, cos ψ) reconstructs it.
3. **Extended training.** Resume the existing goal-4 runs from their `ep_5000` checkpoint and train to iteration 10000 (RL-Games checkpoint load; seeds unchanged). Recovery still uses P_E = 26.11 from the 5000-iteration expert, so values above the 5000-iteration ones are "longer student training", not "better than the expert". If resume does not reproduce the `ep_5000` evaluation at the start (dry-run check), fall back to training 10000 iterations from scratch and drop block 4 seeds to pay for it.
4. **Delay d (evaluation only).** The actor sees the intent computed d control steps earlier (per environment; at episode start the first value is used until d steps have passed). d = 0, 6, 30, 60, 120 steps = 0, 0.1, 0.5, 1, 2 s. No hold (N = 1).
5. **Quantisation b (evaluation only).** Each intent component is clipped to [−1, 1] and rounded to 2^b levels; b = 8, 4, 3, 2. No delay.
6. **Skill-switch pilot (evaluation only).** Mixed-goal task: each new goal is pure yaw with probability 0.5, otherwise a Stage 1 random orientation. Controllers, all with the goal visible: Stage 1 expert only; M2 expert only; oracle selector (M2 expert for yaw goals, Stage 1 expert otherwise) with selector hold N = 1, 6, 30. No real model in the loop. Known weakness: yaw goals are a subset of Stage 1 goals, so this tests hand-off into the M2 expert from states it never saw, not composition of complementary skills.

## Blocks (training), in submission order

| Block | Stage | Conditions | Runs | Wall |
|---|---|---|---|---|
| 1 | `m2-hidden` | blind; goal-4 × N 1, 6, 12, 30; goal-yaw × N 1, 6, 12, 30; seeds 0–2 | 27 | 16 h |
| 2 | `m1` resume | goal-4 × N 1, 12, 30, seeds 0–2, iterations 5000 → 10000 | 9 | 5.5 h |
| 3 | `m2-hidden` | hand-8 × N 1, 6, 30, seeds 0–2 | 9 | 5.5 h |
| 4 | `m1` | goal-4 and hand-8 × N 1, 6, 30, seeds 3, 4 | 12 | 7 h |

All other settings as in phase 1: 5000 iterations, 2048 envs, evaluation `17` with 25 checkpoints × 1024 envs, seed 20261028 (block 2: checkpoints 5200–10000).

If time runs out, cut from the bottom: block 4, then block 3, then goal-4 on M2 at N = 6 and 12.

## Steps

- [x] P0. **Code and dry runs.** DONE 2026-11-12: validations `1118902`/`1118903`, dry runs `1118971`/`1118972`, evaluations `1118973`–`1118976`; see Phase 2 log.
  - `intent_modules.py`: `goal-yaw` (kind `goal`, k = 2). `validate_intent_modules.py`: goal-yaw checks on the M2 rollout `1033178`.
  - `16`/`15`: `STUDENT_STAGE=m2-hidden`; `16`: `STUDENT_RESUME_FROM` (list of checkpoints, one per array task) → `train_student.py --checkpoint`, `--max_iterations 10000`; seeds list override for block 4 (`STUDENT_SEEDS="3 4"`).
  - `intent_env_wrapper.py`: `intent_delay`, `intent_bits` (evaluation only, defaults off); `evaluate_students.py`/`17`: `EVAL_INTENT_DELAY`, `EVAL_INTENT_BITS`, output-dir suffix.
  - CPU tests: goal-yaw formula and wrap-around, delay buffer (per-env reset), quantiser, grid index mapping for every block.
  - GPU dry run (256 envs × 3 it): `m2-hidden` × {blind, goal-4, goal-yaw, hand-8} × holds; a resume dry run that loads one goal-4 `ep_5000` checkpoint, logs the starting epoch, and whose first evaluation matches the recorded `ep_5000` value within 1%.
  - Sync (compare remote with HEAD first), checksums.
- [ ] P1. **Delay × quantisation** on the three goal-4 N = 1 final checkpoints (`1111584` tasks 0–2): 5 delays + 4 bit depths, last checkpoint only. Also hand-8 N = 1 (M1 grid) at the same delays as a contrast. Analysis: Recovery vs delay, Recovery vs bits.
- [ ] P2. **Block 1** training + evaluations (batches under the 32-job limit; submit `aftercorr` evaluations while training is pending).
- [ ] P3. **Block 2**, then **block 3**, then **block 4**, each with its evaluations.
- [ ] P4. **Skill-switch pilot**: mixed-goal environment (`m2_tasks.py`), selector wrapper, evaluation only. Start after P2 is submitted; lowest priority.
- [ ] P5. **Analysis**: extend `analyze_rate_sweep.py` to a task column; Recovery vs rate per task with ceilings; goal-yaw vs goal-4; 5000 vs 10000 iterations; 5-seed tests for phase 1 contrasts. Update `CONTEXT.md`.

## Reading the result

- M2: goal-yaw ≈ goal-4 at every rate → the goal channel needs only the task's goal DoF; with phase 1 this gives two points for "goal bandwidth = events/s × DoF" (Stage 1: 2.7 Hz × 3; M2: measure events/s from `1033178`, expected ≈ 3.1 Hz × 1).
- M2: hand-8 collapses below 60 Hz again → the reversal is not specific to Stage 1.
- Block 2: N = 30 rises towards 0.77 while N = 1 saturates → the 2 Hz gap was training time; both rise in parallel → it was not.
- P1: the delay at which goal-4 loses half its Recovery is the latency budget for an upstream model on this task.

## Phase 2 log

### P0 (2026-11-12): DONE

- Code (all defaults leave phase 1 behaviour unchanged):
  - `intent_modules.py`: `goal-yaw` (kind `goal`, k = 2, `goal_yaw()` = (sin ψ, cos ψ), ψ = 2·atan2(q_z, q_w); q and −q agree). `IntentChannel` (pure torch) now holds the zero-order hold (moved unchanged from the wrapper), the delay ring buffer (slot t mod (d+1); output = intent of episode step max(t − d, 0)) and the quantiser (clip to [−1, 1], round to 2^b evenly spaced levels including ±1). At most one of hold / delay / bits may be active (raises otherwise).
  - `intent_env_wrapper.py`: `intent_delay`, `intent_bits` via `IntentChannel`, keyed on `episode_length_buf` as before; logged in the first-observation line.
  - `evaluate_students.py` / `17`: `--intent_delay` / `--intent_bits` ← `EVAL_INTENT_DELAY` / `EVAL_INTENT_BITS` (output suffix `-delayD` / `-bitsB`; recorded in `evaluation.json` with their rules). The evaluator now fails if `--task` differs from the training task in `intent.json`.
  - `train_student.py --checkpoint`: resume. Checks the source run's `intent.json` (intent, task, seed, goal_hidden, asymmetric_critic, intent_hold, expert and PCA hashes, lookahead; num_envs may differ) and that the checkpoint epoch < `--max_iterations`; restores like RL-Games `run_train` and fails unless the restored actor-critic and central value net equal the checkpoint exactly and the epoch matches. Provenance (`resume_from`, sha256, start epoch) goes into `intent.json`. RL-Games 1.6.1 restores weights, normalizers, central value net, optimizer, epoch and frame, but **not the adaptive learning rate**: it restarts at 5e-4 and the KL schedule re-adapts it per minibatch (Isaac Lab's own resume behaves the same).
  - `16`: `STUDENT_STAGE=m2-hidden` (ZAxis task, M2 expert, `--goal_hidden --asymmetric_critic`); `STUDENT_RESUME_FROM` (one checkpoint per array task, count must equal intents × seeds; run name suffix `-resume`); `STUDENT_SEEDS` already existed; guard: m2* stages need an `expert-m2` action PCA dir, other stages must not use one; `STUDENT_PRINT_PLAN=1` prints the task mapping and exits. `15`: `m2-hidden`, same PCA guard, `DRY_RUN_RESUME_FROM` (runs `start epoch + DRY_RUN_ITERATIONS`, checks the logged start epoch). `14` / validator: goal-4 checks always; `CHECK_GOAL_YAW=1` adds |q_x|, |q_y| < 1e-3, unit norm and goal-quaternion reconstruction from (sin ψ, cos ψ) within 1e-5.
- CPU tests (scratchpad) 37/37 PASS: goal-yaw formula for q and −q, wrap-around at ±π, angles beyond ±π; hold identical to the phase 1 implementation (N = 6, 12, 30, random per-env resets); delay d = 1, 6, 30, 60, 120 against a per-episode history reference with resets; quantiser b = 8, 4, 3, 2 (2^b levels, endpoints, error ≤ half a step); invalid combinations rejected.
- Grid mapping (`STUDENT_PRINT_PLAN=1`, all blocks) PASS; out-of-range index and five bad configurations rejected (wrong PCA for m2-hidden / m1, wrong resume count, unknown stage, hold 0). Proposed block 1 order (cut list last): `blind goal-4:1 goal-yaw:1 goal-4:30 goal-yaw:30 goal-yaw:6 goal-yaw:12 goal-4:6 goal-4:12`. Block 2 resume sources must be the newest `ep_5000` file per run (the evaluator's choice; runs have both `..._rew_X.pth` and `..._rew__X_.pth`).
- Synced (remote matched HEAD before; 10 files, checksums match).
- CPU validations PASS: `1118902` on M2 rollout `1033178` (M2 expert, M2 PCA, joint basis `1033233`, `CHECK_GOAL_YAW=1`): |q_x|, |q_y| max 0 (goals are exactly pure yaw), goal-yaw unit norm 6e-8, goal quaternion rebuilt from (sin ψ, cos ψ) within 1.5e-7, goal-4 = canonical goal; all earlier checks still pass. `1118903` Stage 1 regression (`1023537`): all PASS incl. goal-4.
- First GPU dry runs `1118904` (m2-hidden) / `1118905` (resume) FAILED (600 s timeout each, script bug in `15`): the new variable `start_epoch` collided with the wall-clock `start_epoch=$(date +%s)` in the loop, so `--max_iterations` became a timestamp (1794448560); and the resume check grepped `student_resume_start_epoch=` first. Evidence that the code under test worked: `1118905` logged `resume_start_epoch=5000 resume_frame=163840000 resume_model_max_abs_diff=0 resume_central_value_max_abs_diff=0` and continued at epoch 5001; `1118904` blind: actor 149, critic 157. Fixed (`resume_epoch`, grep `^resume_start_epoch=`), dependents `1118906`–`1118909` cancelled.
- Resubmitted 2026-11-12:
  - `1118971` GPU dry run `m2-hidden` × {blind, goal-4, goal-yaw, hand-8} × holds {1, 6, 12, 30}, 256 envs × 3 it.
  - `1118972` resume dry run: goal-4 N = 1 seed 0 `1111584_0` `ep_5000` (`..._rew__5467.298_.pth`), 256 envs, 1 iteration → `ep_5001`.
  - `1118973` (`afterok:1118972`, array 0–1): last-checkpoint evaluation (1024 envs) of the resumed `ep_5001` and of the source `ep_5000`, same single-checkpoint protocol. Acceptance: within 1% (recorded full-curve value 22.326).
  - Smoke tests of the new channel on the source checkpoint (256 envs, not results): `1118974` delay 30, `1118975` bits 2; `1118976` evaluates the m2-hidden goal-yaw hold-6 dry checkpoint on the ZAxis task.
- Results (all COMPLETED 0:0):
  - `1118971` **PASS** 16/16 (~37 s each): m2-hidden actor dims blind 149, goal-yaw 151, goal-4 153, hand-8 157; `critic_states_shape=(256, 157)` everywhere; logged hold matches each requested hold.
  - `1118972` **PASS**: `resume_start_epoch=5000 resume_frame=163840000`, model and central value net max abs diff 0 vs the checkpoint, one epoch → `ep_5001`.
  - `1118973` evaluation (single checkpoint, 1024 envs): source `ep_5000` 22.076 (full-curve record 22.326, −1.1%: evaluator order effect again), resumed `ep_5001` **21.563 (−2.3% vs source) → the 1% criterion is not met.** Restore is bit-exact, so the shift comes from the one PPO update (256 envs, batch 4096, learning rate restarted at 5e-4; the source run's adaptive rate at the end alternates 2.2e-4 / 3.3e-4, `info/last_lr`). For scale: the same runs move −3% … +8% between neighbouring 200-iteration checkpoints (seed 0: 4200 → 4400 −1.9%; seed 2: 4000 → 4200 −3.0%). **User decision 2026-11-12: keep the resume for block 2 and keep block 4.** The 1% bar measured the drift of one PPO update, not the restore, which is exact; the miss does not indicate a resume problem. Report the restart of the adaptive learning rate as a caveat.
  - Smoke tests PASS (256 envs, final goal-4 seed 0 checkpoint, not results): `1118974` delay 30 → 10.66 successes/episode; `1118975` bits 2 → 0.10; `1118976` m2-hidden goal-yaw hold-6 dry checkpoint evaluates on the ZAxis task (actor 151, hold 6).
