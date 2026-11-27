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

Budget: about 40 h wall clock on the cluster, 4 GPUs (3 training + 1 evaluation; raised to 8 GPUs on 2026-11-12, see log), ≤ 32 queued jobs. 57 training runs × ~1:47 = 19 waves ≈ 34 h, plus queueing and reruns. Target: everything finished by 2026-11-14 morning; demo 2026-11-18. Tick each step off with its job ID, as in phase 1.

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
3. **Extended training.** Resume the existing goal-4 runs from their `ep_5000` checkpoint and train to iteration 10000 (RL-Games checkpoint load; seeds unchanged). Recovery still uses P_E = 26.11 from the 5000-iteration expert, so values above the 5000-iteration ones are "longer student training", not "better than the expert". Resume acceptance (user decision 2026-11-12; replaces the original "first evaluation within 1% of `ep_5000`" check, which measured one PPO update and sits below the evaluator's own repeatability of about 1%):
   - (a) Restore: restored actor-critic and central value net equal the checkpoint exactly and the start epoch is 5000. Enforced by `train_student.py --checkpoint` in every resumed run; dry run `1118972` PASS.
   - (b) Transient: over the 9 resumed runs, the mean of P(5200), P(5400), P(5600) relative to the same run's phase 1 `ep_5000` evaluation, averaged over runs, must not be more than 3% below 1. Checked in P5 from the block 2 evaluations.
   - If (b) fails: train goal-4 for 10000 iterations from scratch and drop block 4 to pay for it (the original fallback). Report the restart of the adaptive learning rate (5e-4, one or two schedule steps above the 2.2–3.3e-4 at the end of the source runs) as a caveat either way.
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
  - GPU dry run (256 envs × 3 it): `m2-hidden` × {blind, goal-4, goal-yaw, hand-8} × holds; a resume dry run that loads one goal-4 `ep_5000` checkpoint, logs the starting epoch, and whose first evaluation matches the recorded `ep_5000` value within 1% (criterion replaced 2026-11-12, see definition 3).
  - Sync (compare remote with HEAD first), checksums.
- [x] P1. **Delay × quantisation** DONE 2026-11-12: `1119092` (42/42 PASS), see Phase 2 log. on the three goal-4 N = 1 final checkpoints (`1111584` tasks 0–2): 5 delays + 4 bit depths, last checkpoint only. Also hand-8 N = 1 (M1 grid) at the same delays as a contrast. Analysis: Recovery vs delay, Recovery vs bits.
- [ ] P2. **Block 1** training + evaluations (batches under the 32-job limit; submit `aftercorr` evaluations while training is pending).
- [ ] P3. **Block 2**, then **block 3**, then **block 4**, each with its evaluations.
- [ ] P4. **Skill-switch pilot**: mixed-goal environment (`m2_tasks.py`), selector wrapper, evaluation only. Start after P2 is submitted; lowest priority.
- [ ] P5. **Analysis**: block 2 transient check first (definition 3 (b): checkpoints 5200–5600 vs each run's phase 1 `ep_5000` evaluation); extend `analyze_rate_sweep.py` to a task column; Recovery vs rate per task with ceilings; goal-yaw vs goal-4; 5000 vs 10000 iterations; 5-seed tests for phase 1 contrasts. Update `CONTEXT.md`.

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
  - `1118973` evaluation (single checkpoint, 1024 envs): source `ep_5000` 22.076 (full-curve record 22.326, −1.1%: evaluator order effect again), resumed `ep_5001` **21.563 (−2.3% vs source) → the 1% criterion is not met.** Restore is bit-exact, so the shift comes from the one PPO update (256 envs, batch 4096, learning rate restarted at 5e-4; the source run's adaptive rate at the end alternates 2.2e-4 / 3.3e-4, `info/last_lr`). For scale: the same runs move −3% … +8% between neighbouring 200-iteration checkpoints (seed 0: 4200 → 4400 −1.9%; seed 2: 4000 → 4200 −3.0%). **User decision 2026-11-12: keep the resume for block 2 and keep block 4.** The 1% bar measured the drift of one PPO update, not the restore, which is exact; the miss does not indicate a resume problem. Report the restart of the adaptive learning rate as a caveat. New acceptance criterion (restore exact + transient ≤ 3% over checkpoints 5200–5600): definition 3.
  - Smoke tests PASS (256 envs, final goal-4 seed 0 checkpoint, not results): `1118974` delay 30 → 10.66 successes/episode; `1118975` bits 2 → 0.10; `1118976` m2-hidden goal-yaw hold-6 dry checkpoint evaluates on the ZAxis task (actor 151, hold 6).

### P1 + P2 submission (2026-11-12 10:35)

- Code: `23_eval_channel_sweep.sbatch` runs `17` sequentially once per (condition, run) with `EVAL_LAST_ONLY=1` in one job (one GPU, one queue slot instead of 42); `CHANNEL_PLAN` = `<base|delayD|bitsB>@<run indices>`, plan validated before any evaluation, job fails if any evaluation fails. `submit_block_evals.sh` = generalised `submit_rate_sweep_b2_evals.sh`: submits evaluations of a training array in chunks under the 32-job limit (finished tasks without dependency, others `aftercorr`, each chunk `afterany` on the previous one), state file `eval_state_<train job>.txt`. `analyze_channel_sweep.py` (P1 analysis; checks 42 single `ep_5000` evaluations, N = 1, one of delay/bits). Local tests: wrapper with a stub `17` (env per condition, failure count, bad plan rejected); analysis on synthetic data (PASS; missing file → FAIL). Synced, checksums match; all other synced files still match HEAD.
- **P1** `1119092`: runs (`~/cluster_smoke/p1_channel_sweep_dirs.txt`) 0–2 = goal-4 N = 1 `1111584_0-2`, 3–5 = hand-8 N = 1 `1104420_9-11`; plan `base, delay6, delay30, delay60, delay120 @ 0–5; bits8, bits4, bits3, bits2 @ 0–2` = 42 evaluations, ~66 s each. Output dirs `runs/eval-<run>[-delayD|-bitsB]-job1119092_<i>`. First attempt `1119087` cancelled after 8 s: `sbatch --export=…,CHANNEL_PLAN=base@0,1,…` split the plan at the commas (only `base@0` arrived); it left one partial directory `runs/eval-student-m1-goal-4-seed0-job1119087_0` (ignore). Resubmitted with the variables exported in the shell and `--export=ALL`.
- **P2 block 1** training `1119088` (`--array=0-26%3`; `STUDENT_STAGE=m2-hidden`, PCA `action-pca-expert-m2-rollout-seed0-job1033178-job1033179`, 5000 it; intents list `~/cluster_smoke/phase2_block1_intents.txt`, task i → entry i//3, seed i%3). Headers of tasks 0–2 checked: ZAxis task, blind, expert `35dc6eb1…`, M2 PCA, hold 1. Eval-dir list `~/cluster_smoke/phase2_block1_eval_dirs_1119088.txt` (27 entries). Evaluations 0–3 `1119094` (`aftercorr:1119088,afterany:1119092`, `EVAL_TASK=Isaac-Repose-Cube-Shadow-ZAxis-Direct-v0`). Queue 32/32 at submission; remaining evaluations 4–26 are submitted by `submit_block_evals.sh` as room frees (a Mac-side loop runs it every 20 min; if it stops, run `EXTRA_DEP= ./submit_block_evals.sh 1119088 phase2_block1_eval_dirs_1119088.txt 0 26 Isaac-Repose-Cube-Shadow-ZAxis-Direct-v0 submit` by hand). ETA training ≈ 9 waves × 1:47 ≈ 16 h (≈ 02:30 11-13).

### P1 (2026-11-12): DONE, `1119092` COMPLETED 0:0 (45:46), 42/42 `student_eval_job_status=PASS`, `channel_sweep_analysis_status=PASS`

- Evaluation copies `Docs/Milestone/data/p1_channel_evals/` (42 files, `<run>[-delayD|-bitsB]-job1119092_<i>.json`); output `Experiment Plan/results/rate-sweep-p1-channel/` (summary.json, runs.csv, recovery_vs_delay_bits.png). Baselines (same protocol, last checkpoint) reproduce the grid: goal-4 0.766 vs 0.771, hand-8 0.865 vs 0.866.

Recovery (final checkpoint, mean ± sd, seeds 0–2), retained = P / P(baseline, same run), falls in brackets; N = 1, evaluation only:

| Delay | 0 | 0.1 s (6) | 0.5 s (30) | 1 s (60) | 2 s (120) |
|---|---|---|---|---|---|
| goal-4 | 0.766 ± 0.072 [8%] | 0.642 ± 0.052 (0.84) [6%] | 0.387 ± 0.020 (0.51) [3%] | 0.260 ± 0.008 (0.34) [3%] | 0.157 ± 0.005 (0.21) [3%] |
| hand-8 | 0.865 ± 0.113 [3%] | 0.003 (0.004) [91%] | 0.000 [97%] | 0.000 [97%] | 0.000 [93%] |

| goal-4 bits per component | float | 8 | 4 | 3 | 2 |
|---|---|---|---|---|---|
| Recovery | 0.766 | 0.766 ± 0.074 | 0.120 ± 0.035 | 0.013 | 0.003 |

Findings:

1. **hand-8 does not tolerate any delay**: 0.1 s (6 steps) already drops it to 0.003 with 91% falls. The action-level intent is only usable when it is computed from the current state.
2. **goal-4 degrades smoothly with delay**: 84% retained at 0.1 s, 51% at 0.5 s, 34% at 1 s, 21% at 2 s. Latency budget (first tested delay with ≤ half the baseline) = 1 s; half is crossed between 0.5 and 1 s. The students were trained without delay, so this is a lower bound.
3. **goal-4 needs > 4 bits per component**: 8 bits is lossless (0.766), 4 bits keeps 16%, ≤ 3 bits nothing. 4 bits = step 2/15 ≈ 0.13 per quaternion component, the same order as the 0.1 rad success tolerance, so the policy reaches the quantised goal, not the true one. Reading: the goal must be sent with roughly the precision of the success tolerance; 4 × 8 = 32 bits per goal suffices.

### Block 2 prepared for auto-submission (2026-11-12 11:46)

- Block 1 evaluation 4 submitted by the loop: `1119457` (`aftercorr:1119088,afterany:1119094`).
- Block 2 = `STUDENT_STAGE=m1`, PCA `job1024096`, `STUDENT_ITERATIONS=10000`, intents `goal-4:1 goal-4:30 goal-4:12` × seeds 0–2 (60 Hz vs 2 Hz first), task i ↔ resume source i. Sources = newest `ep_5000` file per run (the evaluator's rule; every run has two): `1111584_0–2` (N = 1), `1116466_24–26` (N = 30), `1116466_36–38` (N = 12); list `~/cluster_smoke/phase2_block2_resume_from.txt`. Print-plan of tasks 0–8 checked: intent, hold and seed of each source equal the target; names `student-m1-goal-4[-holdN]-seed<s>-resume`; seed 0 source = the file the resume dry run `1118972` used.
- Evaluation needs no change: a resumed run's `nn/` holds only its own checkpoints (5200–10000, every 200 → 25), and `17` evaluates all of them (Stage 1 task).
- `submit_block2_auto.sh 1119088 submit` (idempotent): submits training `--array=0-8%3 --dependency=afterany:1119088` once the queue has room for 9 (job ID → `phase2_block2_train_job.txt`, eval-dir list `phase2_block2_eval_dirs_<job>.txt`); submits evaluations via `submit_block_evals.sh` only after all 27 block 1 evaluations are submitted, chained `afterany` on the last block 1 evaluation array (one evaluation GPU). Dry run on the login node: `NO_ROOM_TRAINING` at 32/32, as expected. Estimated: room for block 2 training after ~6 block 1 waves (≈ 21:00), training starts when block 1 training ends (≈ 02:30 11-13), ≈ 3 waves × 1:47 ≈ 5.5 h.
- A Mac-side loop runs the block 1 evaluation helper and then `submit_block2_auto.sh` every 20 min, until `BLOCK2_DONE`. If it stops, run both by hand in `~/cluster_smoke`: `./submit_block_evals.sh 1119088 phase2_block1_eval_dirs_1119088.txt 0 26 Isaac-Repose-Cube-Shadow-ZAxis-Direct-v0 submit; ./submit_block2_auto.sh 1119088 submit`.

### GPU limit raised to 8 (user decision 2026-11-12 12:00)

- The cluster QOS allows 8 GPUs per user (`MaxTRESPU gres/gpu=8`), 16 running and 32 submitted jobs (association `MaxJobs=16 MaxSubmit=32`); the xgpf pool has 14 T4 (10 busy at 11:55, 7 of them another user's). User: use 8 instead of the planned 4.
- Scheme: training arrays throttle %7 and `nice=100`, evaluations nice 0 and %1 (chained), so a freed GPU goes to a waiting evaluation first; the QOS caps the total at 8. Applied to block 1: `scontrol update JobId=1119088 ArrayTaskThrottle=7`, `Nice=100` → tasks 3–6 started 11:58 on xgpf4/5/7/8 (7 training running). New block 1 ETA ≈ 4 waves ≈ 18:30–19:00 (was 02:30 11-13).
- `submit_block2_auto.sh`: block 2 training now `--array=0-8%7 --nice=100` without `afterany:1119088` (block 1 tasks are older, so they still go first); it still waits for 9 free queue slots. The 32-submission limit is now the bottleneck, not GPUs.

### Block 2 submitted; resume criterion replaced (verified 2026-11-12 17:40)

- **Decision (user)**: resume kept for block 2, block 4 kept, acceptance criterion replaced by definition 3 (a) restore + (b) transient. Nothing was resubmitted for this.
- **Block 2 training `1121503`** (submitted 16:30 by `submit_block2_auto.sh`, `--array=0-8%7 --nice=100`): tasks 0–2 RUNNING, 3–8 pending on the 8-GPU cap. Headers of tasks 0–1 checked: Stage 1 task, goal-4, hold 1, 2048 envs, expert `b9b7…`, PCA `job1024096`, `resume_from` = the listed `ep_5000` files of `1111584_0/1`. Task 0 at epoch 7715/10000, checkpoints 5200–7600 written. Eval-dir list `~/cluster_smoke/phase2_block2_eval_dirs_1121503.txt`.
- Block 2 evaluations submitted so far (`eval_state_1121503.txt`): task 0 `1121506`, 1–2 `1121672`, 3–6 `1122027` (`aftercorr:1121503`, chained); **7–8 not submitted yet** (`submit_block2_auto.sh 1119088 submit` adds them when the queue has room).
- Block 1 at the same time: training `1119088` 0–22 COMPLETED, 23–26 RUNNING; all 27 evaluations submitted (`1119094`, `1119457`, `1119856`, `1120007`, `1120096`, `1120409`, `1120513`, `1120634`, `1121494`), 0–18 COMPLETED, 19 running. PASS lines not yet read for block 1.
- Cluster copies of every script used by these jobs equal the local files (md5, 17:40). `23_eval_channel_sweep.sbatch`, `analyze_channel_sweep.py`, `submit_block_evals.sh`, `submit_block2_auto.sh` are still untracked in Git.


### Status 2026-11-12 19:36: block 1 DONE, block 2 running, block 3 SUBMITTED

- **Block 1 DONE**: training `1119088` 27/27 COMPLETED; evaluations (`1119094`, `1119457`, `1119856`, `1120007`, `1120096`, `1120409`, `1120513`, `1120634`, `1121494`) 27/27 COMPLETED, `student_eval_job_status=PASS`. Analysis pending (P5).
- **Block 2**: training `1121503` submitted 16:30 (Mac loop SSH failed 15:26–16:09, so later than possible). Tasks 0–2 COMPLETED/PASS, 3–8 running. All 9 logs: `resume_start_epoch=5000 resume_frame=163840000`, model and central value max abs diff 0. Evaluations all submitted (`1121506`, `1121672`, `1122027`, `1122196`, chained %1); loop ended `ALL_SUBMITTED`.
- Only 7 GPUs were in use at 19:36 (block 2 had 6 training tasks left), so **block 3 was submitted** to fill the 8th: training `1122700` (`--array=0-8%7 --nice=100`, `m2-hidden`, M2 PCA `job1033179`, 5000 it, intents `hand-8:1 hand-8:30 hand-8:6` × seeds 0–2, list `phase2_block3_intents.txt`). Task 0 header checked (ZAxis task, hand-8, expert `35dc6eb1…`, M2 PCA, hold 1). Evaluations 0–7 `1122702` (`aftercorr:1122700,afterany:1122196`, ZAxis task); task 8 goes in through a Mac loop (`submit_block_evals.sh 1122700 phase2_block3_eval_dirs_1122700.txt 0 8 Isaac-Repose-Cube-Shadow-ZAxis-Direct-v0 submit`). Running: 8 GPUs.

### Block 1 analysis (2026-11-12 20:00): `phase2_block1_analysis_status=PASS`

- `cluster_smoke/analyze_phase2_block1.py` on the 27 evaluations (copies `Docs/Milestone/data/phase2_block1_evals/`, named `<run>-job<eval>_<i>.json`); output `Experiment Plan/results/phase2-block1-m2/` (summary.json, runs.csv, m2_recovery_vs_rate.png). Checks: ZAxis task for training and evaluation, goal hidden, 25 checkpoints ending at 5000, evaluated hold = trained hold, no delay/bits, 3 seeds per condition.
- P_E = 31.137 (M2 expert `1032490`, ep_5000). **P_B = 19.66** (blind seeds 0–2: 20.80 / 16.32 / 21.86): the M2 actor without any goal information already reaches 63% of the expert, with large seed spread. Recovery on M2 therefore divides by P_E − P_B = 11.5 and is noisy; it exceeds 1 when a student beats the expert. Successes per episode are reported alongside.

| Condition | 60 Hz (N=1) | 10 Hz (N=6) | 5 Hz (N=12) | 2 Hz (N=30) |
|---|---|---|---|---|
| goal-4: successes / Recovery | 32.70 / 1.14 ± 0.06 | 28.92 / 0.81 ± 0.15 (p = 0.05) | 26.84 / 0.63 ± 0.11 (p = 0.004) | 16.29 / −0.29 ± 0.06 (p < 0.001) |
| goal-yaw: successes / Recovery | 35.66 / 1.39 ± 0.15 | 31.25 / 1.01 ± 0.29 (p = 0.14) | 28.20 / 0.74 ± 0.20 (p = 0.01) | 16.22 / −0.30 ± 0.10 (p < 0.001) |
| goal-yaw vs goal-4, p | 0.08 | 0.36 | 0.43 | 0.93 |
| ceiling (600/N)/P_E, successes cap | 1 | 1 | 1 | 0.64 (20 successes) |
| Stage 1 goal-4 Recovery (phase 1) | 0.77 | 0.71 | 0.65 | 0.43 |

p in the rows = Welch vs the same intent at N = 1 (uncorrected). Falls ≤ 3% everywhere. N80 (0.8 P_E) reached by 3/3 seeds of every goal condition at N ≤ 12, by none at N = 30 or blind.

Findings:

1. **goal-yaw ≈ goal-4 at every rate** (p ≥ 0.08; goal-yaw nominally higher at 60–5 Hz). On a 1-DoF goal task, 2 numbers carry what 4 do (question 2).
2. **At 60–5 Hz the goal intent beats the M2 expert's level or comes close** (goal-yaw 60 Hz 35.7 vs expert 31.1), and degrades with hold as on Stage 1.
3. **At 2 Hz both goal intents fall below blind** (16.3 vs 19.7 successes). The refresh cap is 20 successes per episode at N = 30; blind is not capped and reaches 19.7. Hypothesis (not verified): on pure-yaw goals the blind policy turns the cube continuously and passes each target, so a stale goal is worse than none. A video of a blind run would confirm.
4. Consequence for question 1: the M2 reversal test needs hand-8 (block 3) compared in successes, not Recovery, and against blind as a reference, since P_B is high and noisy.

### M2 blind behaviour check: SUBMITTED 2026-11-12 ~20:10 (user approved video)

- Purpose: test the hypothesis from block 1 finding 3 (blind turns the cube continuously and passes the yaw goals).
- `record_videos.py` now also writes `<name>_trace.npz` (cube and goal quaternion, success counter, episode step; all envs, every control step); video and success outputs unchanged. Remote matched HEAD before sync; checksums match.
- `analyze_m2_video_traces.py`: per run |yaw rate|, direction consistency (|Σ Δψ| / Σ |Δψ|), turns per episode, share of still steps (< 0.5 rad/s), tilt, successes per episode. Synthetic test: steady 2 rad/s spin → consistency 1.00, 3.2 turns, still 0.00; stop-and-go → 0.20, 0.3, 0.50.
- Job `1122889` (one GPU, runs `21` sequentially, 16 envs, 600 steps, ZAxis task, final checkpoints): blind seeds 0–2 (`1119088_0–2`), goal-yaw N = 1 seed 0 (`_6`), goal-yaw N = 30 seed 0 (`_12`). Run list `~/cluster_smoke/phase2_m2_video_dirs.txt`; outputs `~/data/low-dim-intent/videos/<run>-job1122889_<i>/`.

### M2 blind behaviour check: DONE 2026-11-12 20:58, `1122889` COMPLETED 0:0 (10:48), `video_seq_status=PASS` (5/5)

- Copies (videos, traces, json): `Docs/Milestone/data/m2_video_traces_1122889/` (not for Git); metrics `Experiment Plan/results/phase2-block1-m2/video_traces_1122889.json` (`analyze_m2_video_traces.py`, 16 envs × 600 steps per run, final checkpoints).

| Run | |yaw rate| (rad/s) | direction consistency | turns / episode | still steps | tilt | successes / episode |
|---|---|---|---|---|---|---|
| blind seed 0 | 7.1 | 0.99 | 11.3 | 5% | 2.0° | 21.8 |
| blind seed 1 | 5.4 | 0.99 | 8.5 | 6% | 1.9° | 17.9 |
| blind seed 2 | 7.1 | 1.00 | 11.3 | 2% | 1.7° | 23.4 |
| goal-yaw 60 Hz seed 0 | 6.8 | 0.06 | 0.6 | 5% | 9.9° | 36.6 |
| goal-yaw 2 Hz seed 0 | 4.4 | 0.09 | 0.3 | 31% | 9.7° | 22.0 |

- **Hypothesis confirmed**: the blind M2 policy spins the cube about z in one direction (consistency ≥ 0.99, 9–11 turns per episode, almost never still) and collects ≈ 2 successes per turn (a new uniform yaw goal is on average half a turn ahead). The goal-informed policies turn back and forth (consistency < 0.1) towards each goal; at 2 Hz the goal-yaw policy is still 31% of the time (waiting on the stale goal).
- Consequence: on M2 a goal-free strategy reaches 63% of the expert; the goal interface only adds value at refresh rates ≥ 5 Hz. At 2 Hz the cap (one goal per refresh, 20 / episode) makes a goal interface worse than spinning. For the report: M2's P_B is a "spin" strategy, not a failure baseline; compare M2 conditions in successes against blind.
- Note: the 16-env traces are a small sample (goal-yaw 2 Hz 22.0 here vs 16.2 in the 1024-env evaluation); use them for behaviour, not for performance numbers.

### Evaluations on 2 GPUs (user decision 2026-11-12 21:05)

- Split of the 8-GPU cap changed from 7 training + 1 evaluation to **6 training + 2 evaluations**: `scontrol update ArrayTaskThrottle=2` on evaluation arrays `1122196`, `1122702`, `1122784`; `ArrayTaskThrottle=6` on block 3 training `1122700`. Arrays stay chained (`afterany`), so at most 2 evaluations run. Running jobs are not preempted: the 7th block 3 training task finishes normally and its slot then goes to an evaluation (nice 0 vs 100). Evaluations are independent single-GPU jobs with fixed seeds, so running them in parallel does not change their results.
- `submit_block_evals.sh`: array throttle now `%${EVAL_PARALLEL:-2}` (was `%1`), for block 4. Synced, checksums match.
- Block 2 training `1121503` 9/9 finished by 21:05; video job `1122889` done.

### Block 4 SUBMITTED 2026-11-12 ~22:20 (user request, before block 3 drains)

- Only 6–7 of 8 GPUs were in use. `scontrol update JobId=1122700_8 Nice=0` was accepted (user may lower nice) and task 8 started at once.
- **Block 4 training `1123488`** (`--array=0-11%6`, submitted `--nice=100`, then `Nice=0`): `STUDENT_STAGE=m1`, PCA `job1024096`, 5000 it, `STUDENT_SEEDS="3 4"`, intents `goal-4:1 goal-4:6 goal-4:30 hand-8:1 hand-8:6 hand-8:30` (list `~/cluster_smoke/phase2_block4_intents.txt`). Print-plan of tasks 0–11 checked: Stage 1 task, names `student-m1-<intent>[-holdN]-seed{3,4}`, same naming as the phase 1 seeds 0–2. Eval-dir list `phase2_block4_eval_dirs_1123488.txt`.
- Evaluations 0–9 `1123489` (`aftercorr:1123488,afterany:1122784`, Stage 1 task, %2). **Tasks 10–11 not yet submitted** (queue at 32/32): `EXTRA_DEP=afterany:1122784 ./submit_block_evals.sh 1123488 phase2_block4_eval_dirs_1123488.txt 0 11 Isaac-Repose-Cube-Shadow-Direct-v0 submit` once the queue has room.
- At 22:20 block 4 was PENDING (Priority): other users' fairshare priority (13k–61k) exceeds ours (≈6.4k). Nice 0 on block 4 means a free slot may go to a block 4 training rather than a block 3 evaluation when both wait; the QOS still caps the total at 8.
- Caveat: resume check (b) (definition 3) is still open. If it fails, the fallback drops block 4; cancel `1123488`/`1123489` then.

### Resume check (b): PASS (2026-11-12 ~22:40)

- `cluster_smoke/analyze_resume_transient.py` on the 9 block 2 evaluations (`1121506_0`, `1121672_1–2`, `1122027_3–6`, `1122196_7–8`, all `student_evaluation_status=PASS`, 25 checkpoints 5200–10000 each; copies `Docs/Milestone/data/phase2_block2_evals/`) against the same runs' phase 1 evaluations (`Docs/Milestone/data/rate_sweep_grid_evals/`). Output `Experiment Plan/results/phase2-block2-resume/resume_transient.json`. Checks: same task, goal hidden, hold, evaluation seed, episodes, success rule; each `resume_from` is exactly the `ep_5000` file the phase 1 evaluation used (9/9).

| Run | P(5000) phase 1 | P(5200) / P(5400) / P(5600) | ratio | P(10000) |
|---|---|---|---|---|
| goal-4 N=1 s0 | 22.33 | 23.03 / 22.73 / 23.15 | 1.029 | 25.47 |
| goal-4 N=1 s1 | 18.43 | 18.72 / 18.61 / 19.27 | 1.024 | 21.77 |
| goal-4 N=1 s2 | 19.65 | 19.70 / 20.41 / 20.93 | 1.035 | 23.73 |
| goal-4 N=12 s0 | 17.22 | 17.63 / 17.71 / 17.90 | 1.030 | 20.09 |
| goal-4 N=12 s1 | 16.40 | 16.38 / 16.46 / 16.34 | 1.000 | 19.26 |
| goal-4 N=12 s2 | 17.12 | 17.68 / 17.66 / 17.98 | 1.038 | 20.45 |
| goal-4 N=30 s0 | 10.36 | 10.97 / 10.38 / 10.96 | 1.040 | 11.68 |
| goal-4 N=30 s1 | 11.81 | 11.84 / 11.72 / 11.72 | 0.996 | 12.50 |
| goal-4 N=30 s2 | 11.18 | 11.27 / 11.58 / 11.48 | 1.023 | 12.33 |

- **Mean ratio 1.024 (criterion ≥ 0.97) → `resume_transient_status=PASS`.** No run dips: the lowest is 0.996. With (a) already verified on all 9 logs, the resume is accepted; block 4 (`1123488`) stays. The −2.3% of the 256-env one-update dry run does not appear in the 2048-env runs. The adaptive-LR restart remains a report caveat.
- P(10000) is the last checkpoint only; the 5000 vs 10000 comparison belongs to the block 2 analysis (P5).

### Block 2 analysis: 5000 vs 10000 iterations (2026-11-12 ~23:00), `phase2_block2_analysis_status=PASS`

- `cluster_smoke/analyze_phase2_block2.py` (same inputs as the resume check; each resumed run joined to its own phase 1 curve, 200–10000). Output `Experiment Plan/results/phase2-block2-resume/` (`summary.json`, `m1_goal4_5000_vs_10000.png`). M1 constants P_E = 26.11 (5000-iteration expert), P_B = 0.006; Recovery at 10000 = longer student training, not "better than expert".

| goal-4, M1 | Recovery @5000 | Recovery @10000 | Δ (paired p) | share of N=1 @5000 → @10000 | ceiling |
|---|---|---|---|---|---|
| 60 Hz (N=1) | 0.771 ± 0.076 | 0.906 ± 0.071 | +0.135 (0.006) | 1 → 1 | 1 |
| 5 Hz (N=12) | 0.648 ± 0.017 | 0.763 ± 0.023 | +0.116 (0.003) | 0.84 → 0.84 | 1 |
| 2 Hz (N=30) | 0.426 ± 0.028 | 0.466 ± 0.017 | +0.040 (0.03) | 0.55 → 0.51 | 0.77 |

Findings:

1. **Students were still improving at 5000 iterations**: doubling training adds +0.12–0.14 Recovery at 60 and 5 Hz (all 6 runs up). The 60 Hz student reaches 0.91 of the expert.
2. **The rate penalty is not an under-training artefact.** The 5 Hz / 60 Hz ratio stays at 0.84; the 2 Hz run plateaus at 0.47 (+0.04 over 5000 iterations), far below its refresh ceiling 0.77, so the gap to 60 Hz widens (0.55 → 0.51 of N=1; p = 0.006 at 10000).
3. Phase 1 conclusions at 5000 iterations stand; absolute Recovery values at 5000 are lower bounds for fast rates.

### Video recorder bug fixed; demo clips SUBMITTED (2026-11-12 ~22:45)

- **Bug**: `record_videos.py` built `IntentRlGamesVecEnvWrapper` without `intent_hold`, so every video played the policy with a fresh intent each step (60 Hz), whatever hold it was trained under. Affected: held runs only. In `1122889` that is clip 4, goal-yaw N = 30 seed 0: its row in the trace table (22.0 successes, 31% still) shows a 2 Hz-trained policy played at 60 Hz, not 2 Hz behaviour (the 1024-env evaluation, which applies the hold, gives 16.2). The blind and N = 1 rows, and the spin conclusion, are unaffected. Fix: pass `intent_hold` from `intent.json` (as `evaluate_students.py` does) and record it in the video json. Synced, checksum match.
- New `24_demo_videos.sbatch`: runs `21` sequentially on one GPU over `~/cluster_smoke/phase2_demo_video_list.txt` (`<task> <run dir>` per line), 16 envs, final checkpoints. Job **`1123641`** (submitted 22:43 when the queue had room): 0 M1 hand-8 N=6 seed 0 (`1111584_6`), 1 M1 goal-4 N=1 seed 0 (`1111584_0`), 2 M1 goal-4 N=30 seed 0 (`1116466_24`), 3 M2 goal-yaw N=30 seed 0 re-recorded with the fix (`1119088_12`), 4 M2 hand-8 N=1 seed 0 (block 3, `1122700_0`). Outputs `videos/<run>-job1123641_0/`.

### Staleness vs Recovery scatter (2026-11-12, CPU, existing data): `staleness_scatter_status=PASS`

- `cluster_smoke/analyze_staleness_scatter.py`: step 0 offline hold metrics (`results/intent-hold-job1110496/`) against retained Recovery = R(N)/R(1) of the 39-run grid (`results/rate-sweep-grid-39runs/runs.csv`), 12 points (4 intents × N = 6, 12, 30). Output `results/rate-sweep-staleness/` (summary.json, points.csv, retained_vs_staleness.png).
- Normalised hold error predicts retained Recovery better than the update rate: Spearman −0.86 (p = 0.0003) vs −0.50 (p = 0.10); leave-one-intent-out RMSE of exp(−a·x) 0.23 vs 0.46.
- It is not a single law: obj-4 and goal-4 have the same hold error (0.24 / 0.49 / 1.13 vs 0.22 / 0.49 / 1.17) but retain 0.66 / 0.40 / 0.06 vs 0.92 / 0.84 / 0.55. The share of steps outside the 0.1 rad tolerance separates them (obj-4 0.66–0.91, goal-4 0.11–0.55; Spearman −0.94, n = 6).
- Caveats: 12 points in three clusters (hand-k all at error 1.65–1.91), so the correlation is mostly between-cluster; the tolerance metric exists only for quaternion intents; no test of the fit on a second task.
- 23:25 (user): block 4 had started 6 training tasks at nice 0 and filled the 8-GPU cap with block 3's last 2, blocking block 3 evaluations and demo clips `1123641` (`QOSMaxGRESPerUser`). `scontrol update JobId=1123488 ArrayTaskThrottle=4`; the 6 running tasks continue, new block 4 tasks start only while < 4 run.
- Demo clips `1123641` COMPLETED 23:59 (`demo_videos_status=PASS`, 5/5; outputs `videos/<run>-job1123641_0/`). Recorded holds = trained holds (6, 1, 30, 30, 1). Re-recorded M2 goal-yaw N=30 seed 0: 16.9 successes per env (16 envs, 600 steps) vs 16.2 in the 1024-env evaluation, so the fix holds; the old `1122889` clip-4 row (22.0) is superseded.

### Block 3 analysis: M2 hand-8 (2026-11-13), `phase2_block3_analysis_status=PASS`

- Training `1122700_0–8` COMPLETED 0:0; evaluations `1122702_0–7`, `1122784_8` COMPLETED, 9/9 `student_evaluation_status=PASS`. Copies `Docs/Milestone/data/phase2_block3_evals/` (`<run>-job<eval>_<i>.json`).
- `cluster_smoke/analyze_phase2_block3.py` (reuses the block 1 loader and checks; blind/goal comparators from the 27 block 1 evaluations). Output `Experiment Plan/results/phase2-block3-m2-hand8/` (summary.json, runs.csv, m2_hand8_vs_rate.png). Checks: ZAxis task for training and evaluation, goal hidden, 25 checkpoints ending at 5000, evaluated hold = trained hold, 3 seeds per condition. P_E 31.14, P_B 19.66 as in block 1.

| M2, goal hidden: successes ± sd [Recovery] | 60 Hz (N=1) | 10 Hz (N=6) | 2 Hz (N=30) |
|---|---|---|---|
| hand-8 (seeds 0/1/2) | **36.72 ± 4.69** (31.98 / 36.84 / 41.35) [1.49] | 25.75 ± 2.90 (22.40 / 27.33 / 27.51) [0.53] | **11.67 ± 0.99** [−0.70] |
| goal-4 | 32.70 ± 0.73 [1.14] | 28.92 ± 1.74 [0.81] | 16.29 ± 0.72 [−0.29] |
| goal-yaw | 35.66 ± 1.68 [1.39] | 31.25 ± 3.34 [1.01] | 16.22 ± 1.16 [−0.30] |
| hand-8 vs goal-4 / goal-yaw, p (Recovery) | 0.27 / 0.74 | 0.20 / 0.10 | **0.004 / 0.007** |
| hand-8 vs blind 19.66, p (successes) | 0.010 (above) | 0.06 (above) | **0.03 (below)** |
| hand-8 vs own N = 1, p | – | 0.035 | 0.009 |
| hand-8 falls (mean) | 1.4% | 0.6% | 1.2% |
| hand-8 retained R(N)/R(1): M2 / Stage 1 | 1 / 1 | 0.36 / 0.21 | < 0 / 0.02 |
| hand-8 successes kept S(N)/S(1): M2; goal-4 M2 | 1; 1 | 0.70; 0.88 | 0.32; 0.50 |

N80 (0.8 P_E): hand-8 3/3 at 60 Hz, 2/3 at 10 Hz, 0/3 at 2 Hz. Refresh cap at N = 30 is 20 successes.

Findings (question 1):

1. **The reversal repeats on M2.** At 60 Hz hand-8 is level with the goal intents (36.7 vs 32.7 / 35.7, p ≥ 0.27; nominally the highest, with the largest seed spread). At 10 Hz it is behind both goal intents (25.8 vs 28.9 / 31.3; not significant with 3 seeds, p 0.20 / 0.10). At 2 Hz it is the worst interface: 11.7 successes, below goal-4 and goal-yaw (p ≤ 0.007) and below the goal-free blind spin (p = 0.03).
2. **hand-8 degrades more gently on M2 than on Stage 1.** It keeps 36% of its Recovery at 10 Hz (Stage 1: 21%) and still 83% of P_E in successes. Hypothesis, not verified: M2 motion (yaw turning) is more repetitive, so a held action code goes stale more slowly. The 10 Hz gap to the goal intents is therefore a trend, not a significant result.
3. **No falls collapse.** Students trained under the hold fall in ≤ 2% of episodes at every rate (as in Stage 1 trained runs); the 2 Hz loss is fewer successes, not drops.
4. Caveats: 3 seeds, uncorrected tests; no N = 12 for hand-8 (plan); M2 Recovery divides by P_E − P_B = 11.5 and exaggerates differences, so successes are the primary measure.

Presentation: `Docs/Final presentation/make_figures.py` rerun (M2 figure now has the hand-8 line; label "hand-8 (no 5 Hz run)"); script slide 10 placeholder filled. Deck not rebuilt yet (rebuild together with the demo video).

### Block 4: training DONE, evaluations 10–11 SUBMITTED (2026-11-13 ~04:15)

- Training `1123488_0–11` all COMPLETED 0:0 (last two ended 04:10). Evaluations 0–9 `1123489` COMPLETED, 10/10 `student_evaluation_status=PASS`.
- Evaluations 10–11: **`1125291`** (`submit_block_evals.sh 1123488 phase2_block4_eval_dirs_1123488.txt 0 11 Isaac-Repose-Cube-Shadow-Direct-v0 submit`, no dependency: both trainings finished, queue empty; `EXTRA_DEP=afterany:1122784` dropped since `1122784` is done). Dry run first: done=10 11, nothing else pending. Next: block 4 analysis (5-seed tests of the phase 1 contrasts) once `1125291` passes.

---

# Phase 3 (2026-11-13, overnight): goal sent on change, arriving late — students trained under delay

User request 2026-11-13 ~04:20: run something overnight from the "next steps" of the talk (action sequences, event-triggered goals, a real slow planner). Chosen by the agent, **to be confirmed by the user**: only the event-triggered goal channel is feasible now. An action sequence needs the expert's future actions (a forked simulation per refresh); a real planner needs model access from the compute nodes and has nothing to decide on this task (the goal is given).

## Definition (scientific change)

- **Training under delay d.** Same channel as definition 4 of phase 2 (the actor sees the intent computed d control steps earlier in the episode; the episode's first value until d steps have passed), now also during training. For goal-4 this is an event-triggered channel with latency d: the goal value changes only on a success, and the actor learns of each change d steps late. No hold (N = 1). The critic still sees the current goal. Everything else as in M1 (Stage 1 task, goal hidden, 5000 iterations, 2048 envs, seeds 0–2, evaluation `17`, 25 checkpoints × 1024 envs).
- Evaluation uses the trained delay: `evaluate_students.py` now defaults `--intent_delay` to the run's `intent.json` value (as for hold) and reports `intent_delay_trained`.

## Conditions (12 runs)

goal-4 × d = 6, 30, 60 (0.1, 0.5, 1 s); hand-8 × d = 6 (0.1 s); seeds 0–2.

## Questions and reading

1. Waiting-time model (agent's check on summary means, 2026-11-13, not yet per seed): if a delayed goal only costs d steps per goal, Recovery = 600 / (T0 + d) / P_E with T0 = 600 / (R(0)·P_E). It reproduces the evaluation-only delays of P1 (0.638 / 0.383 / 0.255 / 0.153 predicted vs 0.642 / 0.387 / 0.260 / 0.157) and the holds at 10 and 5 Hz (0.711 / 0.651 vs 0.707 / 0.648, mean wait (N−1)/2), but not 2 Hz (0.519 vs 0.426). Do students trained under delay stay on this line (0.64 / 0.38 / 0.26 for d = 6 / 30 / 60 from R(0) = 0.77), fall below it, or beat it?
   - On the line → latency costs time only; the "latency budget" is arithmetic of the task's goal rate.
   - Above the line → the student uses the waiting time (the model is too pessimistic). Below → delay also hurts learning.
2. hand-8 at 0.1 s: evaluation-only gave 0.003 with 91% falls. Does training under the delay remove the falls and recover anything (as training under hold did: 0.03 → 0.18 at 10 Hz)?

## Log

### Code and submission (2026-11-13 04:30)

- Code: `train_student.py --intent_delay` (written to `intent.json`, in the resume match keys, default 0); `16`: intent entry `intent:dD` → delay D, run name `-delayD`; `15`: `DRY_RUN_DELAY`, checks the logged delay; `evaluate_students.py`: default delay = trained delay; `record_videos.py`: plays with the trained delay. Local checks: py_compile, `bash -n`, print-plan mapping of the 12 tasks. Remote matched local before the sync; checksums match after.
- Dry run **`1125326`** (`15`, m1, goal-4 and hand-8, delay 30, 256 envs × 3 it).
- Training **`1125327`** (`16`, `--array=0-11%6`, `--dependency=afterok:1125326`), intents `goal-4:d6 goal-4:d30 goal-4:d60 hand-8:d6` (`~/cluster_smoke/phase3_delay_intents.txt`), names `student-m1-<intent>-delayD-seed{0,1,2}`; eval-dir list `phase3_delay_eval_dirs_1125327.txt`.
- Evaluations **`1125328`** (`17`, 0–11%2, `aftercorr:1125327`, Stage 1 task).
- If the dry run fails, `1125327` and `1125328` stay pending (dependency never satisfied): cancel both, fix, resubmit.
- 04:35: dry run `1125326` COMPLETED, `student_dry_run_job_status=PASS` (goal-4 and hand-8 at delay 30, 3 epochs each, obs 153 / 157). Training `1125327` tasks 0–5 RUNNING, 6–11 waiting on the array limit; evaluations `1125328` waiting on their dependency. No training result yet.

### Results check (2026-11-13, preliminary; no analysis script yet)

- Training `1125327_0–11` COMPLETED 0:0 (`student_train_status=PASS` 12/12, last ended 08:04). Evaluations `1125328_0–10` + `1125498_11` COMPLETED, 12/12 `student_evaluation_status=PASS`. Copies `Docs/Milestone/data/phase3_delay_evals/` (`<run>-job<eval>_<i>.json`). Checks (all 12): Stage 1 task, goal hidden, evaluated delay = trained delay, hold 1, action hold 1, no bits, 25 checkpoints ending at 5000.
- Recovery (last checkpoint, M1 P_E/P_B), mean ± sd of 3 seeds, vs evaluation-only delay (P1, `results/rate-sweep-p1-channel/`) and the waiting-time line (R(0) = 0.766):

  | condition | trained under delay | evaluation-only | waiting-time line | Welch p |
  |---|---|---|---|---|
  | goal-4 d6 | 0.628 ± 0.009 (fall 0.06) | 0.642 | 0.638 | 0.69 |
  | goal-4 d30 | 0.257 ± 0.015 (fall 0.04) | 0.387 | 0.383 | 0.001 |
  | goal-4 d60 | 0.188 ± 0.007 (fall 0.05) | 0.260 | 0.255 | 0.0003 |
  | hand-8 d6 | 0.091 ± 0.147 (seeds 0.008 / 0.261 / 0.003; fall 0.04) | 0.003 (fall 0.91) | — | 0.41 |

- Reading (preliminary): question 1 → d6 on the line; d30 and d60 **below** it (≈ −33% and −27% vs evaluation-only), i.e. delay during training also hurts learning, not only time. Caveat: d6 curves still rise (+0.04 to +0.08 Recovery from iteration 4000 to 5000); d30/d60 are flat (−0.03 to +0.04). Question 2 → training removes the falls (0.91 → 0.04) but recovers performance in only 1 of 3 seeds (seed 1, 0.26); seeds 0 and 2 stay near 0 with flat curves.

### Phase 3 analysis (2026-11-13), `phase3_delay_analysis_status=PASS`

- `cluster_smoke/analyze_phase3_delay.py`; inputs `Docs/Milestone/data/phase3_delay_evals/` (evaluations `1125328_0–10`, `1125498_11`), `p1_channel_evals/` (`1119092`), `rate_sweep_grid_evals/` (hold-trained goal-4 / hand-8); output `Experiment Plan/results/phase3-delay/` (summary.json, runs.csv, delay_trained_vs_eval_only.png). Checks: Stage 1 task, goal hidden, evaluated delay = trained delay, hold / action hold / bits off, 25 checkpoints ending at 5000, 3 seeds per condition, P1 rows are 60 Hz students at ep_5000.
- Unit: time per goal T = 600 / successes (control steps). T0 (60 Hz goal-4, no delay) = 30.0 steps (seeds 27.2 / 32.6 / 30.7). Extra = T − T0 (same seed for evaluation-only rows; group-mean T0 for trained rows, which are other networks).

  | goal-4 | Recovery | waiting line | extra steps per goal | wait |
  |---|---|---|---|---|
  | evaluation-only d 6 / 30 / 60 / 120 | 0.642 / 0.387 / 0.260 / 0.157 | 0.638 / 0.382 / 0.255 / 0.153 | 5.8 / 29.4 / 58.2 / 116.0 | 6 / 30 / 60 / 120 |
  | trained d 6 | 0.628 ± 0.009 | 0.638 | 6.6 (6.1–7.2) | 6 |
  | trained d 30 | 0.257 ± 0.015 (p = 0.0013 vs evaluation-only) | 0.383 | 59.5 (56.2–65.7) | 30 |
  | trained d 60 | 0.188 ± 0.007 (p = 0.0003) | 0.255 | 91.9 (87.6–96.4) | 60 |
  | trained hold 6 / 12 / 30 (phase 1) | 0.707 / 0.648 / 0.426 | 0.707 / 0.647 / 0.516 | 2.5 / 5.5 / 24.1 | 2.5 / 5.5 / 14.5 |

Findings:

1. **The waiting-time line holds per seed for evaluation-only delay**: max |Recovery error| 0.007 over 3 seeds × 4 delays; the measured extra time is the delay minus 0.1–6.4 steps (largest at d = 120). A 60 Hz goal-4 student under delay only waits. Half of the Recovery is lost at d ≈ T0 = 0.5 s.
2. **Training under the delay does not help and hurts from 0.5 s on**: d 30 / d 60 lose about d + 30 steps per goal instead of d (Recovery −33% / −28% vs evaluation-only), curves flat from iteration 4000 (change −0.03 to +0.04). d 6 is on the line and still rising (+0.04 to +0.08).
3. **Same pattern under hold**: goal-4 trained at hold 6 / 12 is on the line (extra 2.5 / 5.5 = mean age), hold 30 is not (24.1 vs 14.5). The earlier "2 Hz does not fit" may be the same training effect. Not tested: no evaluation-only hold of the 60 Hz goal-4 students exists (cross-check below).
4. hand-8 d 6: 0.091 ± 0.147 (0.008 / 0.261 / 0.003), falls 0.91 → 0.04 vs evaluation-only; p = 0.41. Training under the delay removes the falls, recovers in 1 of 3 seeds; that seed equals hand-8 trained at hold 6 (0.09–0.30).
- Cause of finding 2 is open (slower skill, or behaviour during the wait). Reading stays within the plan's question 1: "below the line → delay also hurts learning".

### Cross-check evaluations SUBMITTED 2026-11-13 (user approved), evaluation only

- Code: `23_eval_channel_sweep.sbatch` accepts `delay0` (forces no delay on a delay-trained run) and `hold<N>` (sets `EVAL_INTENT_HOLD`); `bash -n`; synced (cluster md5 `97975aa1…`; `17`, evaluator and wrapper unchanged, md5 = local). Last checkpoint, 1024 envs, seed 20261028. Record `~/cluster_smoke/crosscheck_jobs.txt`.
- **`1128340`** (Stage 1 task; dirs `crosscheck_m1_dirs.txt`: 0–8 the goal-4 delay-trained runs d 6 / 30 / 60 × seeds 0–2, 9–11 `student-m1-goal-4-seed{0,1,2}` of `1111584`): plan `delay0@0–8 hold6@9–11 hold12@9–11 hold30@9–11` (18 evaluations).
  - delay0: T back at ≈ 30 steps → the skill is intact and the loss is in the waiting behaviour; T ≈ 58 → training under delay gave a slower controller.
  - evaluation-only hold: on the line at hold 30 (0.52) → the 2 Hz shortfall of the hold-trained students (0.43) is a training effect, as for delay.
- **`1128341`** (Task 2, `EVAL_TASK=Isaac-Repose-Cube-Shadow-ZAxis-Direct-v0`; dirs `crosscheck_m2_dirs.txt`: `student-m2-hidden-goal-yaw-seed{0,1,2}` of `1119088`): plan `base@0–2 delay6@0–2 delay30@0–2 delay60@0–2` (12 evaluations). Out-of-sample test of the waiting line: T0 ≈ 600 / 35.7 ≈ 17 steps, so successes should follow 600 / (T0 + d) per seed, with half lost at ≈ 0.28 s.
- Acceptance: `channel_sweep_job_status=PASS`, per evaluation `student_evaluation_status=PASS`. Submitted only — not a result.

### Cross-check results (2026-11-13, read from per-run JSONs; no analysis script yet)

- `1128340` and `1128341` COMPLETED; 30/30 `student_evaluation_status=PASS`. Copies: `Docs/Milestone/data/crosscheck_1128340/`, `Docs/Milestone/data/crosscheck_m2_delay_1128341/`. Last checkpoint (5000), 1024 episodes. Stage 1 P_E 26.11, T0 per seed 27.18 / 32.60 / 30.70 steps (summary.json).
- **Delay-trained goal-4 evaluated at delay 0** (successes per seed → mean, Recovery, T = 600 / successes):

  | trained d | at own d (`1125328`) | at d = 0 (`1128340`) | Recovery at d = 0 | T at d = 0 |
  |---|---|---|---|---|
  | 0 (60 Hz ref.) | — | 20.01 | 0.766 | 30.0 |
  | 6 | 16.39 | 18.64 / 19.18 / 18.19 → 18.67 | 0.715 | 32.1 |
  | 30 | 6.72 | 10.60 / 8.96 / 10.44 → 10.00 | 0.383 | 60.0 |
  | 60 | 4.93 | 9.84 / 9.26 / 10.03 → 9.71 | 0.372 | 61.8 |

  Reading: d 6 skill intact (+2 steps). **d 30 / d 60: T ≈ 60 even with no delay → training under delay produced a slower controller** (≈ +30 steps per goal), not a waiting-only loss. With delay, T at own d = 89 / 122 ≈ T(d=0) + d (60 + 30, 62 + 60): the delay-trained loss = slower controller + pure waiting. Falls 5–10%.
- **60 Hz goal-4, evaluation-only hold** (waiting (N − 1) / 2; line = 600 / (T0 + wait) per seed):

  | hold N | seeds 0 / 1 / 2 | mean | line | Recovery (line) | trained at same N |
  |---|---|---|---|---|---|
  | 6 (10 Hz) | 20.39 / 17.10 / 18.22 | 18.57 | 18.46 | 0.711 (0.707) | 0.71 (5 seeds) |
  | 12 (5 Hz) | 18.64 / 15.78 / 16.54 | 16.99 | 16.89 | 0.650 (0.647) | — |
  | 30 (2 Hz) | 14.92 / 12.89 / 13.69 | 13.83 | 13.47 | 0.529 (0.516) | 0.44 (5 seeds) |

  Every seed within 0.5 successes of the line. **At 2 Hz the hold-trained students (0.44) are below the 60 Hz students under evaluation-only hold (0.53)** → like delay, training under a slow channel costs more than the waiting itself. Falls 4–8%.
- **Task 2 goal-yaw, evaluation-only delay** (out-of-sample test; T0 per seed 17.02 / 15.81 / 17.56; blind 19.66):

  | d (s) | seeds 0 / 1 / 2 | mean | line | measured wait (steps) |
  |---|---|---|---|---|
  | 0 | 35.25 / 37.96 / 34.17 | 35.79 | — | — |
  | 6 (0.1) | 26.14 / 28.13 / 25.86 | 26.71 | 26.35 | 5.9 / 5.5 / 5.6 |
  | 30 (0.5) | 13.34 / 13.61 / 13.12 | 13.35 | 12.83 | 28.0 / 28.3 / 28.2 |
  | 60 (1.0) | 8.46 / 8.52 / 8.31 | 8.43 | 7.82 | 53.9 / 54.6 / 54.6 |

  The waiting line holds out of sample: measured wait ≈ 0.9 d on every seed (Stage 1: d − 0.3…2), error ≤ 8% of successes. Falls ≤ 1.2%. goal-yaw drops below blind between d 6 and d 30 (line crossing at d ≈ 13 steps ≈ 0.23 s); at 0.5 s it reaches 13.4 vs blind 19.7.

---

# Phase 4 (approved by the user 2026-11-13 ~04:50, overnight): slow expert (action hold)

Why: hand-k under a hold is a 60 Hz bang-bang expert's action, held. A reviewer can say the collapse comes from that expert, not from the action level. This phase asks what a controller that is itself limited to 60/N Hz can reach.

## Definition (scientific change)

- **Action hold N.** A new policy action is applied only when the episode step of the observation it answers is a multiple of N (always at episode start); on the other steps the previously applied action is repeated (`IntentRlGamesVecEnvWrapper.step`). The environment's `act_moving_average` is 1.0 (no smoothing), actions are joint position targets, so a held action is a held target.
- **Slow expert.** Stage 1 task, goal visible, no intent (`blind`), symmetric critic, i.e. the Stage 1 blind student setting, plus action hold N = 6, 12, 30 (10, 5, 2 Hz). 5000 iterations, 2048 envs, seeds 0–2; evaluation `17` under the same action hold (read from `intent.json`), 25 checkpoints × 1024 envs. Reference at N = 1: the Stage 1 blind students and P_E = 26.11.
- Caveats, fixed before the runs: (1) the policy is still queried every step and PPO also updates on the N − 1 of N actions that are never applied, so the result is a lower bound for a controller designed for that rate (no frame skip, same environment-step budget as every other run); (2) the observation does not contain the phase (steps until the next refresh); (3) no refresh ceiling applies here: the goal is visible every step.

## Reading

- Slow expert far below P_E at 10 Hz → the task needs fast feedback; no action-level interface can be slow. The hold collapse of hand-k is not an artefact of the expert.
- Slow expert close to P_E at 10 Hz → the collapse of hand-8 under hold comes from holding a 60 Hz expert's action; narrow the claim, and test hand-8 from the slow expert as a follow-up.

## Log

### Code and submission (2026-11-13 ~05:00)

- Code: `intent_env_wrapper.py` (`action_hold`, `step` override), `train_student.py --action_hold` (in `intent.json` and the resume match keys), `evaluate_students.py` / `record_videos.py` (use the trained action hold), `16` (`STUDENT_ACTION_HOLD`, run name `-aholdN`), `15` (`DRY_RUN_ACTION_HOLD`), new `25_train_and_eval.sbatch` (runs `16` then `17` in one job: one queue entry per run, because the queue was at 24–25 of 32). CPU test of the hold rule incl. an episode reset; py_compile; `bash -n`. Synced, checksums match.
- Note: phase 3 tasks `1125327_6–11` and evaluations `1125328` start after this sync and therefore run the new `train_student.py` / wrapper / evaluator with `action_hold = 1` (no hold; `step` passes through). Tasks 0–5 run the earlier copies.
- Dry run `1125373` (`15`, stage1, blind, action hold 6).

### Block 4 analysis: 5 seeds (2026-11-13), `phase2_block4_analysis_status=PASS`

- Evaluations `1123489_0–9` + `1125291_10–11`, 12/12 `student_evaluation_status=PASS`; copies `Docs/Milestone/data/phase2_block4_evals/`. Seeds 0–2 from the phase 1 grid (`rate_sweep_grid_evals/`) and the M1 grid (hand-8 N = 1, `m1_evals/`).
- `cluster_smoke/analyze_phase2_block4.py`; output `Experiment Plan/results/phase2-block4-5seeds/` (summary.json, runs.csv, m1_5seed_recovery_vs_rate.png). Checks: Stage 1 task, goal hidden, 25 checkpoints ending at 5000, evaluated hold = trained hold, no delay/bits, seeds 0–4 per condition, identical intent record (expert and PCA hashes, PPO size) across seeds.

| Stage 1, Recovery | seeds 0–2 (phase 1) | seeds 3 / 4 | **5 seeds** mean ± sd | falls |
|---|---|---|---|---|
| goal-4 60 Hz | 0.771 | 0.841 / 0.730 | **0.777 ± 0.067** | 6% |
| goal-4 10 Hz | 0.707 | 0.706 / 0.700 | **0.706 ± 0.008** | 7% |
| goal-4 2 Hz | 0.426 | 0.485 / 0.437 | **0.440 ± 0.033** | 6% |
| hand-8 60 Hz | 0.866 | 0.973 / **0.414** | **0.797 ± 0.233** | 3% |
| hand-8 10 Hz | 0.178 | 0.298 / 0.102 | **0.187 ± 0.102** | 2% |
| hand-8 2 Hz | 0.021 | 0.047 / 0.013 | **0.025 ± 0.017** | 4% |

| Contrast (Welch, uncorrected) | 3 seeds | 5 seeds | Mann-Whitney, 5 seeds |
|---|---|---|---|
| hand-8 vs goal-4, 60 Hz | 0.30 | 0.86 (Δ +0.02) | 0.42 |
| hand-8 vs goal-4, 10 Hz | 0.012 | **0.0003** (Δ −0.52) | 0.008 (full separation) |
| hand-8 vs goal-4, 2 Hz | 0.0001 | **< 0.0001** (Δ −0.42) | 0.008 (full separation) |
| goal-4 10 Hz vs 60 Hz | 0.28 | 0.08 | – |
| goal-4 2 Hz vs 60 Hz | 0.009 | 0.0001 | – |
| hand-8 10 Hz / 2 Hz vs 60 Hz | 0.002 / 0.005 | 0.002 / 0.002 | – |

Retained R(N)/R(1), 5 seeds: goal-4 0.91 (10 Hz), 0.57 (2 Hz); hand-8 0.23, 0.03. N80: goal-4 2/5 at 60 Hz, hand-8 3/5 at 60 Hz, none below.

Findings (question 4):

1. **The phase 1 contrasts hold with 5 seeds.** New seeds land inside the old spread for 11 of 12 runs (|z| ≤ 2.1); seed means move by ≤ 0.07 except hand-8 60 Hz. The reversal is stronger, not weaker: at 10 and 2 Hz every goal-4 seed beats every hand-8 seed (rank test at its 5-vs-5 minimum p = 0.008).
2. **At 60 Hz hand-8 and goal-4 stay indistinguishable** (0.80 vs 0.78, p = 0.86). The phase 1 nominal lead of hand-8 (0.87 vs 0.77) shrinks to +0.02.
3. **hand-8 60 Hz seed 4 is a late starter, not a failure**: training `1123488_7` PASS; successes stay ≈ 0 until iteration ~1600–2000 (other seeds start at 400–800) and still rise at 5000 (10.8 successes, Recovery 0.41). Kept in the data (no exclusion rule was set). It makes hand-8 at 60 Hz the noisiest condition (sd 0.23): an action-level intent can also be slower to learn from.
4. goal-4 10 Hz vs 60 Hz: p = 0.08 with 5 seeds (0.28 with 3): 10 Hz keeps 91% of the 60 Hz Recovery; the drop is small and now borderline, not established.
- Dry run `1125373` COMPLETED, `student_dry_run_job_status=PASS` (blind, action hold 6 logged by the wrapper, 3 epochs, obs 157).
- Submitted (`25_train_and_eval.sbatch`, stage1, blind, 5000 it, `gpu-long`, 4 h limit; list `~/cluster_smoke/phase4_action_hold_jobs.txt`): N = 6 **`1125376`** (seeds 0–2), N = 30 **`1125377`** (seeds 0–2), N = 12 **`1125378`** (seeds 0–1). **N = 12 seed 2 not submitted** (queue at 32/32): `STUDENT_ACTION_HOLD=12 sbatch --array=2 --export=ALL 25_train_and_eval.sbatch` with the same exports once there is room. Run names `student-stage1-blind-aholdN-seedS`; evaluations land in `runs/eval-<name>-job<array job>_0`. The evaluation step inside `25` has not been dry-run; if it fails, the training output remains and `17` can be run on it.
- No result yet.
- 11-13: **`1125378_1` (N = 12 seed 1) FAILED** after 1:42 on xgpe2 (Titan RTX): CUDA error 999 at startup, 0 iterations (node fault; the array had been submitted with `--constraint=xgpf|xgpe`). User: T4 only from now on. **Resubmitted as `1127009`** (`--array=1 --constraint=xgpf`, same exports: stage1, blind, `STUDENT_ACTION_HOLD=12`, 5000 it; plan print checked: `student-stage1-blind-ahold12-seed1`). Appended to `phase4_action_hold_jobs.txt`. **N = 12 seed 2 submitted (user OK) as `1127189`** (`--array=2 --constraint=xgpf`, same exports; plan print checked: `student-stage1-blind-ahold12-seed2`). Phase 4 now complete in the queue: 9 runs.
- ~04:48 (user request): queue at the 32-job limit blocked one demo clip. **Cancelled the pending phase 3 evaluation `1125328_11`** (hand-8 d6 seed 2; waiting on training `1125327_11`, which has not started) to free one slot; the demo recording **`1125391`** (`student-m2-blind-seed0`, list `phase2_demo_median_list_c.txt`) took it. Line `11 1125328` removed from `eval_state_1125327.txt` (backup `.bak`). A Mac loop resubmits evaluation 11 when `1125391` leaves the queue: `submit_block_evals.sh 1125327 phase3_delay_eval_dirs_1125327.txt 11 11 Isaac-Repose-Cube-Shadow-Direct-v0 submit` (same script and protocol; dependency `aftercorr:1125327` plus `afterany:1125328`, so still ≤ 2 evaluations at once). The evaluation of task 11 therefore carries a different job ID; check `eval_state_1125327.txt`. No timing loss expected (training task 11 ends ≈ 3 h later).

### Phase 4 results check (2026-11-13, preliminary; 7 of 9 runs, no analysis script yet)

- `1125376_0–2` (N = 6), `1125377_0–2` (N = 30), `1125378_0` (N = 12 seed 0): COMPLETED 0:0, `train_and_eval_status=PASS`. N = 12 seeds 1/2 (`1127009_1`, `1127189_2`) still running. Evaluation copies `Docs/Milestone/data/phase4_action_hold_evals/`. Checks (all 7): Stage 1 task, goal visible, evaluated action hold = trained, intent hold 1, no delay, 25 checkpoints ending at 5000.
- Final successes per episode / P_E (26.11), falls; Stage 1 blind at N = 1 for reference: 0.974 (seeds 0–2), falls ≈ 6%:

  | slow expert | seeds | P / P_E | falls | curve (successes at it 1000 / 3000 / 5000) |
  |---|---|---|---|---|
  | 10 Hz (N = 6) | 0, 1, 2 | 0.076 / 0.150 / 0.127, **0.118 ± 0.038** | 0.11–0.16 | still rising (seed 0: 0.05 / 0.41 / 1.99) |
  | 5 Hz (N = 12) | 0 | 0.001 | 0.08 | flat ≈ 0.02 |
  | 2 Hz (N = 30) | 0, 1, 2 | 0.000 | 0.07–0.22 | flat ≈ 0.01 |

- Reading (per "Reading" above): **far below P_E at 10 Hz → the task needs fast feedback; the hold collapse of hand-k is not an artefact of holding a 60 Hz expert.** A controller that sees the goal but acts at 10 Hz reaches 0.12 of the expert, below the goal-hidden hand-8 student trained at 10 Hz (0.19, whose own control loop still runs at 60 Hz) and far below goal-4 at 2 Hz (0.44). Caveats: lower bound (PPO also updates on the N − 1 unapplied actions, no phase in the observation, same step budget), 10 Hz curves still rising at 5000, N = 12 has 1 of 3 seeds.
- **N = 12 complete (2026-11-13)**: `1127009_1` (seed 1) and `1127189_2` (seed 2) COMPLETED, `student_train_status=PASS` + `student_evaluation_status=PASS`; copies in `phase4_action_hold_evals/`. Successes 0.021 / 0.045 / 0.026 → **P / P_E 0.001 ± 0.001** (3 seeds), falls 0.04–0.17. Phase 4 is 9/9; the 5 Hz row now has 3 seeds and the reading is unchanged.

### Phase 4 analysis: 9 runs (2026-11-14), `phase4_action_hold_analysis_status=PASS`

- `cluster_smoke/analyze_phase4_action_hold.py`; inputs `Docs/Milestone/data/phase4_action_hold_evals/` (`1125376_0–2`, `1125377_0–2`, `1125378_0`, `1127009_1`, `1127189_2`), Stage 1 blind students (`student_evals/blind-seed*`, N = 1), grid / M1 evaluations for hand-8 and goal-4 at the same rates; output `Experiment Plan/results/phase4-slow-expert/` (summary.json, runs.csv, slow_expert_vs_rate.png). Checks: Stage 1 task, blind, goal visible, evaluated action hold = trained, intent hold 1, no delay, 25 checkpoints ending at 5000, 3 seeds per N.

  | action rate | P / P_E, seeds 0 / 1 / 2 | mean ± sd | falls | episodes with ≥ 1 success | gain 4000 → 5000 (successes) | hand-8 student, same rate | goal-4 student, same rate |
  |---|---|---|---|---|---|---|---|
  | 60 Hz (N = 1) | 1.000 / 0.947 / 0.975 | 0.974 ± 0.026 | 7% | 100% | +0.6 / +1.3 / +1.1 | 0.866 (p = 0.23) | 0.771 (p = 0.03) |
  | 10 Hz (N = 6) | 0.076 / 0.150 / 0.127 | 0.118 ± 0.038 | 14% | 87% | +0.96 / +0.43 / +0.65 | 0.178 (p = 0.43) | 0.707 (p = 0.0007) |
  | 5 Hz (N = 12) | 0.001 / 0.002 / 0.001 | 0.001 | 9% | 3% | ≈ 0 | 0.096 (p = 0.04) | 0.648 (p = 0.0002) |
  | 2 Hz (N = 30) | 0.000 / 0.000 / 0.000 | 0.000 | 14% | 1% | 0 | 0.021 (p = 0.16) | 0.426 (p = 0.001) |

  (Welch on successes, 3 seeds, uncorrected; students are goal-hidden, trained under an intent hold at that rate, own loop at 60 Hz.)

Findings:

1. **A controller that sees the goal but acts at 10 Hz reaches 12% of the expert; at 5 and 2 Hz nothing** (≤ 3% of episodes with any success, curves flat from the start). Going from 60 to 10 Hz costs 88%.
2. **The slow expert is never above the hand-8 student at the same rate** (10 Hz 0.12 vs 0.18, n.s.; 5 Hz 0.001 vs 0.10, p = 0.04) and far below goal-4 (p ≤ 0.001 from 10 Hz down). The hold collapse of hand-k is therefore not an artefact of holding a 60 Hz expert's action: slowing the action loop itself is at least as bad. Per "Reading": the task needs fast feedback.
3. **10 Hz is not converged**: all three seeds still gain 0.4–1.0 successes over the last 1000 iterations (seed 0 +48% of its final value, seeds 1 / 2 +11% / +20%). 0.12 is a lower bound in training length as well as in method (PPO updates on the unapplied actions, no phase in the observation).
4. 5 Hz and 2 Hz give no learning signal at all, so they do not separate "impossible at this rate" from "not learnable with this training setup".

### Phase 4: 10 Hz slow expert resumed 5000 → 10000 — SUBMITTED 2026-11-14 (user approved)

- Why: finding 3 (10 Hz not converged). Question: does the 10 Hz slow expert stay at or below the hand-8 student at 10 Hz (0.18), or rise well above it?
- Job **`1138319`** (`25_train_and_eval.sbatch`, `--array=0-2`, `gpu,gpu-long`, 3 h): `STUDENT_STAGE=stage1`, `STUDENT_INTENTS=blind`, `STUDENT_ACTION_HOLD=6`, `STUDENT_ITERATIONS=10000`, `STUDENT_RESUME_FROM` = newest `ep_5000` file of `1125376_0–2` (block 2 rule; list `~/cluster_smoke/phase4_resume_from.txt`). Plan print checked (seed and action hold of each source = target; names `student-stage1-blind-ahold6-seed{0,1,2}-resume`). Cluster code md5 = local. All three running on xgpf1–3 at submission.
- Evaluation inside the job: Stage 1 task, trained action hold, checkpoints 5200–10000 (25), 1024 envs.
- Acceptance: `train_and_eval_status=PASS` per task; resume restore as in block 2 definition 3 (first resumed checkpoints continue the 5000 value). Submitted only — not a result.

### Phase 4 resume results (2026-11-14, read from the evaluation JSONs; not yet in the analysis script)

- `1138319_0–2`: logs show `student_train_status=PASS`, `student_evaluation_status=PASS`, `train_and_eval_status=PASS` (3/3). `sacct` returned "Unknown error 1054" twice (accounting database), so the Slurm state was not read; the queue is empty. Copies `Docs/Milestone/data/phase4_action_hold_resume_evals/`. Checks: Stage 1 task, goal visible, action hold 6 trained and evaluated, intent hold 1, no delay, 25 checkpoints 5200–10000.
- Successes per episode (P / P_E in brackets):

  | seed | 5000 (`1125376`) | 6000 | 8000 | 9000 | 9800 | 10000 | mean 9200–10000 |
  |---|---|---|---|---|---|---|---|
  | 0 | 1.99 (0.076) | 2.81 | 3.67 | 3.84 | 3.66 | 3.62 (0.139) | 0.140 |
  | 1 | 3.92 (0.150) | 4.50 | 4.45 | 4.37 | 4.46 | 4.58 (0.176) | 0.172 |
  | 2 | 3.31 (0.127) | 3.88 | 4.09 | 4.12 | 3.49 | **1.89 (0.072)** | 0.128 |
  | mean | 0.118 ± 0.038 | | | | | 0.129 ± 0.052 | 0.147 |

- **The 10 Hz slow expert plateaus at ≈ 0.13–0.17 of the expert.** Seeds 0 and 1 are flat from iteration ~8000 (3.6–3.8, 4.4–4.6); seed 2 declines from 9000 and halves at the last checkpoint (3.49 → 1.89, falls 7% → 3%, one checkpoint; not investigated). Falls of seeds 0 / 1 stay at 14–20%.
- **vs the hand-8 student at 10 Hz** (goal hidden, 5000 iterations): 0.178 ± 0.105 (3 seeds), 0.187 ± 0.102 (5 seeds). Slow expert at 10000: 0.129 (Welch p = 0.52 vs 3 seeds, 0.33 vs 5 seeds; Mann-Whitney 0.57); with the last-5-checkpoint mean 0.147, best checkpoint per seed 0.163. Not distinguishable, and not above. vs goal-4 at 10 Hz (0.707): p = 0.002.
- Reading: doubling the training does not lift the slow expert above the hand-8 student; finding 2 of the phase 4 analysis stands and finding 3 ("not converged") is closed for 10 Hz: the plateau is ≈ 0.15. Caveats unchanged (no frame skip); the comparison gives the slow expert twice the iterations of the hand-8 student.

### Task 2 (M2) goal-hidden interface sweep, minimal: hand-2 / hand-4 at 60 Hz — SUBMITTED 2026-11-13 (user approved)

- Why: Task 2 goal-hidden at 60 Hz had only blind / goal-4 / goal-yaw / hand-8, so no action-dimension ladder to compare with M1 (hand-2 0.50, hand-4 0.71, hand-8 0.87). Question: does the 1-DoF task need fewer action components?
- Job **`1127361`** (`25_train_and_eval.sbatch`, `--array=0-5`, no throttle; QOS caps the user at 8 GPUs): `STUDENT_STAGE=m2-hidden`, `STUDENT_INTENTS="hand-2 hand-4"` × seeds 0–2 (task i → intent i//3, seed i%3), hold 1, 5000 it, M2 action PCA `job1033179`, `EVAL_TASK=Isaac-Repose-Cube-Shadow-ZAxis-Direct-v0` (1024 envs, seed 20261028, all 25 checkpoints) — same settings as block 3 hand-8. Plan printout checked (names `student-m2-hidden-hand-{2,4}-seed{0,1,2}`); cluster code md5 = local. Record `~/cluster_smoke/m2_hidden_hand24_jobs.txt`.
- Acceptance: per task `train_and_eval_status=PASS`; compare successes per episode with blind 19.66, hand-8 36.72, goal-yaw 35.66 (block 1/3), Welch, 3 seeds. Submitted only — not a result.
- 2026-11-13: pending `1127361`, `1127009`, `1127189` moved to `Partition=gpu,gpu-long`, `TimeLimit=03:00:00` (`scontrol update`). Reason: `25` asks only for `gpu-long`, whose T4s (xgpf6–11, 8 GPUs) were all busy, while xgpf0–5 (`gpu` only, 3 h max) were idle. Train+eval takes 1:45–2:00, so 3 h is enough. Result: 8 GPUs running at once (QOS cap `gres/gpu=8`); `1127361_3–5` wait on QOSMaxGRESPerUser.
- 2026-11-13 (user: free a GPU for demo clips): `1127361_5` (hand-4 seed 2) cancelled after 22 min (iteration 982, nothing kept); clip jobs chained `1127897` → `1127918` (afterany); hand-4 seed 2 resubmitted as **`1127936`** (`--array=5`, same exports, `gpu,gpu-long`, 3 h, `--dependency=after:1127918`, i.e. eligible once the second clip job has started). Run name unchanged (`student-m2-hidden-hand-4-seed2`), output under `-job1127936_5`.
- **Results (2026-11-13)**: `1127361_0–4` + `1127936_5` COMPLETED, 6/6 `student_train_status=PASS` + `student_evaluation_status=PASS` (`1127361_5` cancelled, see above). Copies `Docs/Milestone/data/m2_hidden_hand24_evals/`. Final successes per episode (Welch p, 3 vs 3 seeds):

  | Task 2, 60 Hz | seeds 0 / 1 / 2 | mean | vs blind | vs hand-8 | vs goal-yaw |
  |---|---|---|---|---|---|
  | blind | 20.79 / 16.32 / 21.86 | 19.66 | — | | |
  | hand-2 | 23.67 / 27.10 / 34.36 | 28.38 | p = 0.09 | p = 0.12 | p = 0.14 |
  | hand-4 | 43.62 / 41.65 / 35.01 | 40.09 | p = 0.005 | p = 0.42 | p = 0.23 |
  | hand-8 | 31.98 / 36.84 / 41.35 | 36.72 | | — | |
  | goal-yaw | 35.22 / 37.52 / 34.25 | 35.66 | | | — |

  Reading: **on Task 2, 4 hand components already match hand-8 and goal-yaw** (no significant difference; mean slightly above both); hand-2 sits between blind and hand-8 with a wide seed spread (24–34), not significantly different from either. Compare Stage 1 (hand-2 0.50, hand-4 0.71, hand-8 0.87 Recovery): the 1-DoF task saturates at fewer components. Curves: hand-2 seeds 0/2 and hand-4 seeds 0/1 still rising slowly at 5000 (≤ +1.5 over the last 1000 iterations), so these are lower bounds. Falls ≤ 2%.

---

# Time structure of each action PC (2026-11-14, user request; CPU analysis of the expert rollout)

- Question: do the principal components kept by hand-2 / hand-4 / hand-8 differ in how fast they change, and does any of them carry a slow, goal-level part?
- Code: `cluster_smoke/analyze_pc_timescales.py` + `26_pc_timescales.sbatch` (copy of `22`; reuses `held_index`, `quat_angle` of `analyze_intent_hold.py`). Synthetic test on the Mac (AR(1) signals with known coefficients 0.95…0: lag-1 autocorrelation and slowest direction recovered). Synced, md5 = local.
- Job **`1138991`** (`normal` partition, CPU, 14 s): dataset checksums PASS, `pc_timescales_status=PASS`, `pc_timescales_job_status=PASS`. Rollout `1023537` (1,228,800 steps), PCA `1024096`. Output cluster `datasets/pc-timescales-expert-rollout-seed0-job1023537-job1138991/`; local copy `Experiment Plan/results/pc-timescales-job1138991/` (summary.json, pc_timescales.csv, pc_timescales.png; SHA256SUMS verified). `sacct` not read (accounting error 1054 earlier today).
- Per PC, within episodes (hold error = E[(z − held)²] / var; 1 = as bad as sending the mean):

  | PC | variance | autocorrelation lag 1 / 2 / 3 / 6 | hold error N = 2 / 3 / 6 | goal share (null) |
  |---|---|---|---|---|
  | 1 | 21.9% | 0.68 / 0.22 / −0.13 / −0.18 | 0.32 / 0.74 / 1.60 | 0.04 (0.06) |
  | 2 | 13.9% | 0.49 / −0.03 / −0.29 / −0.02 | 0.51 / 1.03 / 1.79 | 0.06 (0.05) |
  | 3 | 11.5% | 0.51 / 0.04 / −0.21 / −0.01 | 0.48 / 0.96 / 1.68 | 0.10 (0.08) |
  | 4 | 9.5% | 0.41 / −0.06 / −0.18 / 0.01 | 0.58 / 1.09 / 1.65 | 0.07 (0.06) |
  | 5 | 8.1% | 0.53 / 0.10 / −0.17 / −0.05 | 0.47 / 0.91 / 1.66 | 0.08 (0.07) |
  | 6 | 6.0% | 0.54 / 0.08 / −0.16 / 0.01 | 0.46 / 0.92 / 1.62 | 0.10 (0.09) |
  | 7 | 5.3% | 0.43 / 0.05 / −0.01 / 0.00 | 0.57 / 1.01 / 1.50 | 0.09 (0.10) |
  | 8 | 4.8% | 0.50 / 0.06 / −0.15 / −0.08 | 0.50 / 0.96 / 1.65 | 0.05 (0.06) |
  | 9–18 | 1.0–2.9% each | lag 1: 0.29–0.40 | 0.60–0.72 / 0.94–1.16 / 1.33–1.62 | 0.06–0.19 (0.07–0.16) |

  hand-2 / hand-4 / hand-8 as a whole (variance-weighted): lag 1 0.60 / 0.55 / 0.54; lag 4 −0.30 / −0.26 / −0.24; lag 8 +0.11 / +0.13 / +0.12; hold error at N = 3 0.85 / 0.91 / 0.93 (not tabulated for hand-8: from the components), at N = 6 1.67 / 1.67.

Findings:

1. **Every PC forgets itself in about one control step.** Lag-1 autocorrelation 0.41–0.68 for PC 1–8, ≈ 0 at lag 2 (−0.06 to 0.22), negative at lag 3. The autocorrelation falls below 1/e after 1.1–1.7 steps (17–28 ms). Holding any PC for 3 steps is as bad as sending its mean; for 6 steps it is worse (1.5–1.8).
2. **The kept PCs do not differ in time scale.** PC 1 is the slowest (0.68) and also the largest; PC 2–8 are 0.41–0.54; the discarded PC 9–18 are a little faster (0.29–0.40). So hand-2, hand-4 and hand-8 have the same time structure; k changes how much of the action is sent, not how long it stays valid.
3. **(Corrected 2026-11-14, see "Video check" below: not a rhythm.) The action commands alternate over about 4 control steps**: the autocorrelation of hand-k is negative at lag 3–6 (minimum ≈ −0.3 at lag 4) and positive again at lag 8 (+0.12). At a 10 Hz refresh (N = 6) the held value is anti-correlated with the current one, which is why the hold error exceeds 1.
4. **No PC has a goal-level part.** The share of a PC's variance explained by its mean over a goal segment (21 steps on average) equals the null from another environment's segments for every PC (PC 1–8: 0.04–0.10 vs 0.05–0.10; white noise 0.048). The expert's action PCs do not hold a level that depends on the current goal.
5. **No linear direction of the action is slow** (slow feature analysis on the 18 non-degenerate dimensions): the slowest direction has lag-1 autocorrelation 0.72 (19.5% of the action variance, close to PC 1), the next 0.61, 0.59. A different linear basis would not make an action-level intent survive a slow channel.
6. Raw joints look the same (lag 1 0.39–0.59, lag 2 ≈ 0); their distributions are two-valued (excess kurtosis −0.6 to −1.6 for the 13 active joints), the PCs are closer to Gaussian (PC 3, the wrist, −1.1).
- Caveats: one expert (seed 0), near bang-bang commands, linear same-signal statistics.

### Video check of the "7.5 Hz rhythm" (2026-11-14, Mac, existing clips): not confirmed

- Figure `pc_timescales.png` inspected: the three panels match the table (lag-1 bars 0.41–0.68 for PC 1–8, dip at lag 3–6, small bump at lag 8, goal share ≈ null). Cosmetic: the left panel's log axis starts below 0; script fixed (`set_xlim`), figure not regenerated.
- Method: expert clip `clips/student-stage1-blind-seed0-job1125119_0` (checkpoint `b9b7…` = frozen expert; 600 frames at 60 fps = one frame per control step, 10 s), plus the hand-8 and goal-4 60 Hz student clips of `1125119`. Frames → grey, 16×16-pixel block means, the 20% of blocks with the largest variation (hand and cube); Welch spectrum (2 s windows) of the frame-to-frame change. Also the cube's angular speed from the clip trace (16 envs × 600 steps). Scratch script, not in the repository.
- Result: **no peak at 7.5 Hz.** Share of the change spectrum per 1 Hz bin (expert): 5–6 Hz 0.037, 6–7 0.045, 7–8 0.050, 8–9 0.048, 9–13 0.048–0.050, then falling to 0.033 at 19–20 Hz: a flat plateau from 7 to 13 Hz. The position spectrum falls monotonically (0.315 at 1–2 Hz, 0.044 at 7–8, 0.026 at 9–10). The two student clips give the same shape. Cube angular speed: autocorrelation 0.36 / 0.26 / 0.16 / 0.09 at lags 1–4, never negative before lag 7 (−0.02); power concentrated at 2–3.5 Hz (the goal rate).
- Reading: finding 3 was overstated. The commands show a short alternation (autocorrelation −0.24 to −0.30 at lag 4, +0.11 to +0.12 at lag 8, gone by lag 12–20), i.e. a correction that reverses after about 4 steps and dies within one cycle; it is not a sustained rhythm and it does not appear as a periodic motion of the hand or the cube. The statement that survives: a value held for 4–6 steps is anti-correlated with the current command.
- Limits: block brightness is a crude proxy for joint motion; one 10 s clip per policy; joint positions were not analysed (they are in the rollout observations and could be).

### Joint-position time structure: PREPARED 2026-11-14 (user approved), not submitted — SSH failure

- Question: the commands forget themselves in one step; does the hand's measured motion (observation slice `hand_dof_pos`, 24 joints) do the same, or are posture components slow?
- Code: `cluster_smoke/analyze_joint_timescales.py` (reuses `signal_stats` of `analyze_pc_timescales.py`): joints, posture PCs (own PCA of the 24 positions), positions projected on the action PCA directions, slowest linear directions, command PCs for reference; actuated joints matched to commands from the data (lagged correlation, must be one to one). `27_joint_timescales.sbatch` (copy of `26`, lags up to 120). Synthetic test on the Mac PASS (positions built as a low-pass of the commands with a known joint permutation: mapping recovered except that a joint and its exact scaled copy tie).
- **Not submitted**: SSH to `soc-cluster` returned "Connection closed by UNKNOWN port 65535" on 6 attempts from ~14:30 (the connection goes through the jump host `soc-jump`); the two earlier connections of the day worked. Nothing was synced. Layer: SSH/network; no job, no result.


### Joint-position time structure: DONE 2026-11-14, job `1139989`

- SSH worked again at ~16:40. Synced `analyze_joint_timescales.py`, `analyze_pc_timescales.py` (cluster copy differed only by the cosmetic xlim line), `27_joint_timescales.sbatch` to `~/cluster_smoke/` (checksums match). Job **`1139989`** (`normal`, CPU, xcnc0, 63 s): dataset and PCA-basis checksums PASS, `joint_timescales_status=PASS`, `joint_timescales_job_status=PASS`, `failures=[]`. `sacct` still returns error 1054; state read from the log. Output cluster `datasets/joint-timescales-expert-rollout-seed0-job1023537-job1139989/`; local copy `Experiment Plan/results/joint-timescales-job1139989/` (summary.json, joint_timescales.csv, joint_timescales.png; SHA256SUMS verified).
- Results (1,228,800 steps; goal segments 21.0 steps on average; posture PCs 1–8 = 81.1% of the position variance):

| Signal | acf lag 1 | acf lag 6 | acf lag 12 | acf lag 30 | hold error N = 2 / 3 / 6 / 12 / 30 (1 = the mean) |
|---|---|---|---|---|---|
| command PCs 1–8 | 0.41–0.68 | −0.18 to 0.01 | −0.09 to −0.03 | 0.00 | 0.32–0.58 / – / 1.50–1.79 / 1.71–1.81 / 1.88–1.95 |
| posture PCs 1–8 | 0.85–0.94 | 0.16–0.50 | −0.04 to 0.21 | −0.02 to 0.07 | 0.06–0.14 / 0.15–0.36 / 0.44–0.96 / 0.86–1.38 / 1.45–2.65 |
| positions along the action PCA directions 1–8 | 0.87–0.94 | 0.14–0.49 | −0.06 to 0.22 | 0.01–0.06 | 0.05–0.11 / – / 0.46–0.74 / 0.87–1.39 / 1.45–2.50 |

  1. **The hand's posture is smoother than the commands, but only over a few steps.** The posture PC autocorrelation falls below 0.5 after 2.4–6.0 steps (40–100 ms) and is gone by lag 12–30. No sign of the commands' reversal at lag 4.
  2. **Holding a posture value works at 30 Hz, half works at 10 Hz, fails from 5 Hz down.** Hold error at N = 2: 0.06–0.14; N = 6: 0.44–0.96 (commands 1.5–1.8); N = 12: 0.86–1.38; N = 30: worse than the mean for every PC.
  3. **No goal-level component in posture either.** Share of variance explained by the per-goal mean equals the unaligned null for every posture PC (PC1–8: 0.28–0.50 vs null 0.30–0.49).
  4. Slowest linear directions: lag-1 acf 0.985 / 0.976 / 0.967 / 0.963, each 5.5–8.8% of the position variance; the slowest is almost only RFJ3. Single joints: lag-1 acf 0.69–0.97; joint 8 (MFJ2) is the only one with acf 0.39 at lag 30.
- Reading: the answer to the question is "smoother, not slow". A posture-level intent would tolerate about 20–30 Hz where the action intent tolerates none, but it has no component on the goal time scale (21 steps), so it does not give a 2–10 Hz interface. Diagnostic only; no intent definition changed.
- **Caveats on the command-to-joint matching** (affects only the row "positions along the action PCA directions"; posture PCs, single joints and slow features use all 24 positions and do not depend on it):
  - MFJ1, RFJ1, LFJ1 were matched to articulation indices 18, 19, 22 (the coupled distal joints MFJ0, RFJ0, LFJ0) instead of the documented 13, 14, 20; the documented joint is the runner-up each time (0.69 vs 0.67, 0.70 vs 0.67, 0.62 vs 0.59). The match is one to one, so the script passed. The other 17 agree with `[0..16, 20, 21, 23]`.
  - RFJ3 and MFJ2 barely follow their commands (best lagged correlation 0.11 and 0.03; all others 0.52–0.70). Not investigated.
  - Not rerun with the documented index map.

### PCA over time windows (user request 2026-11-14): DONE, job `1141204`

- Question: the per-step PCA needs 14 dimensions and every component forgets itself in a few steps. Is a stretch of motion (L control steps) nevertheless made of a few recurring patterns ("primitives")?
- Code: `cluster_smoke/analyze_window_pca.py` + `28_window_pca.sbatch` (copy of `27`, CPU, no PCA basis). Signals: 20 commands, 24 joint positions. Windows of L = 1, 3, 6, 12, 21, 30 steps inside one episode, cut two ways: sliding (consecutive, non-overlapping) and goal (starting at each goal change). Per set of windows: PCA of the L × D vectors (components for 80 / 90 / 95%, cumulative variance at k); two nulls from the same data, "independent" (time steps unrelated) and "separable" (spatial covariance × one temporal correlation: smoothness only, no order between joint patterns); template share (variance of the mean window); k-means (k = 4, 16, 64; 40,000 windows, seed 0) vs Gaussian windows with the same covariance.
- Synthetic test on the Mac PASS: three planted 12-step patterns give 2 components at L = 12 (nulls 23 and 171) and k = 4 explains 98.5% (null 63%); white-noise commands equal both nulls.
- First submission attempt failed at the SSH layer (jump host, 4 attempts ~17:30); submitted at 20:47 when the connection returned. Job **`1141204`** (`normal`, CPU, xcnz5, 134 s): dataset checksums PASS, `window_pca_status=PASS`, `window_pca_job_status=PASS`, `failures=[]`. Cluster copies of script and sbatch match local (sha256 `015d2dba…`, `de8186d3…`). Output cluster `datasets/window-pca-expert-rollout-seed0-job1023537-job1141204/`; local copy `Experiment Plan/results/window-pca-job1141204/` (summary.json, window_pca.csv, window_pca.png; SHA256SUMS verified).
- Results (components for 95% of the variance; goal segments: mean and median 21 steps):

| Signal, windows | L | actual | null: unrelated steps | null: smoothness only | variance in 8 components | mean window (template) |
|---|---|---|---|---|---|---|
| commands, sliding | 12 | 145 | 167 | 141 | 37% | 0% |
| commands, sliding | 21 | 250 | 292 | 244 | 29% | 0% |
| commands, goal | 12 | 132 | 160 | 142 | 47% | 10% |
| commands, goal | 21 | 237 | 283 | 247 | 34% | 7% |
| positions, sliding | 12 | 61 | 175 | 70 | 55% | 0% |
| positions, sliding | 21 | 100 | 305 | 117 | 44% | 0% |
| positions, goal | 12 | 53 | 162 | 73 | 61% | 9% |
| positions, goal | 21 | 92 | 291 | 122 | 48% | 6% |

  k-means, goal-aligned windows, share of variance explained (Gaussian null in brackets): positions L = 12: k = 4 26% (17%), 16 42% (29%), 64 53% (38%); positions L = 21: 18% (13%), 33% (22%), 43% (29%); commands L = 12: 18% (14%), 36% (23%), 49% (29%); commands L = 21: 12% (9%), 24% (15%), 34% (20%).

  1. **A stride is not a few patterns.** One goal segment (21 steps) of posture needs 92–100 linear components for 95% (32–36 for 80%); 8 components hold 44–48%. Commands need 237–250.
  2. **Commands have no structure in time beyond their one-step correlation**: actual = smoothness-only null (145 vs 141, 250 vs 244), about 12 components per step.
  3. **Posture is compressible in time, mostly because it is smooth**: 61 vs 175 at L = 12 (about 5 components per step instead of 15). Beyond smoothness there is a modest extra structure, larger when windows start at a goal change: 53 vs 73 (L = 12), 92 vs 122 (L = 21), i.e. 25–27% fewer components than smoothness alone (13–15% for sliding windows).
  4. **No small dictionary.** 64 cluster centres explain 43–53% of goal-aligned posture windows; 4 centres 18–26%. Above the Gaussian null by a factor 1.4–1.5, so the windows are somewhat clustered, but far from a handful of discrete primitives.
  5. The stereotyped response to a goal change (mean window) is 6–10% of the variance.
- Reading: in joint space, linearly or by centroids, this expert's motion over one goal is not a combination of a few primitives. There is some repeatable order after a goal change, small next to the variation.
- Limits: (a) Task 1 goals are random 3-D rotations, so windows mix strides about different axes; primitives parametrised by the rotation axis would look high-dimensional here. The cleaner test is the same analysis on the Task 2 rollout (`1033178`, one axis), not run. (b) Fixed-length windows aligned to goal changes, not to contact events; no time warping. (c) Linear PCA and k-means in raw joint coordinates; a learned nonlinear code could be smaller. (d) The excess of k-means over the Gaussian null for commands partly reflects their saturation (62% of entries at a limit), not patterns in time. (e) 48% of the L = 21 goal windows run past the end of their goal segment.

### Block 3b: M2 10 Hz seeds 3–4 (user approved 2026-11-18) — stopping rule fixed before any result

- Purpose: the M2 (`m2-hidden`) 10 Hz comparison hand-8 vs goal-yaw is a trend with seeds 0–2 (25.7 ± 2.9 vs 31.3 ± 3.3 successes/episode, Welch p = 0.099; `results/phase2-block3-m2-hand8/summary.json`). Bring the M2 10 Hz cell to 5 seeds, like Stage 1 block 4.
- Runs: `m2-hidden`, intents `goal-yaw:6 hand-8:6 goal-4:6` × seeds 3–4 (6 runs), 5000 it, 2048 envs, M2 PCA `action-pca-expert-m2-rollout-seed0-job1033178-job1033179`, same scripts as blocks 1/3 (checksums match local). Evaluation: `17`, ZAxis task, last checkpoint protocol as block 1/3.
- **Pre-registered rule (written before training started):** report n = 5 per group (seeds 0–4) for all three intents at 10 Hz, Welch two-sided t-test on final successes/episode, whatever the outcome. No further seeds are added for this comparison, and no seed is dropped (a training failure is rerun with the same seed once; if it fails again it is reported as missing). The n = 3 numbers stay in the record. Report wording: p < 0.05 → "the 10 Hz reversal also holds on M2"; otherwise → "on M2 the reversal is clear at 2 Hz; at 10 Hz it is a trend, milder than Stage 1".
- Acceptance: 6/6 training COMPLETED with `iterations=5000`, 6/6 evaluations `student_evaluation_status=PASS`.
- **SUBMITTED 2026-11-18**: training `1160178` (`--array=0-5%6`, nice 0; tasks 0–1 goal-yaw:6 s3/s4, 2–3 hand-8:6 s3/s4, 4–5 goal-4:6 s3/s4; list `~/cluster_smoke/phase2_block3b_intents.txt`), all 6 started at once on xgpf0–5. Log headers checked: ZAxis task, expert `35dc6eb1…`, M2 PCA `job1033179`, hold 6, 5000 it. Evaluations `1160184_0–5` (`aftercorr:1160178`, throttle raised to 6; eval dirs `phase2_block3b_eval_dirs_1160178.txt`). ETA training ≈ 1 h 50 min, evaluations ≈ 20 min after. Next: check `sacct`, PASS lines, then rerun `analyze_phase2_block3.py` (or a small extension) with seeds 0–4 at 10 Hz under the rule above.
- **RESULT (read 2026-11-19)**: 6/6 training `1160178_0–5` COMPLETED with `student_job_status=PASS` (5000 it); 6/6 evaluations `1160184_0–5` COMPLETED with `student_evaluation_status=PASS` and `student_eval_job_status=PASS`. Final ep_5000 successes/episode (falls): `goal-yaw:6` s3 30.57 (2.1%), s4 30.90 (0.5%); `hand-8:6` s3 16.81 (1.7%), s4 21.25 (1.5%); `goal-4:6` s3 26.15 (1.2%), s4 26.92 (1.8%). Combined with seeds 0–2 from blocks 1 & 3 ($n = 5$, $P_E = 31.14, P_B = 19.66$):
  - `goal-yaw:6` (10 Hz): **31.05 ± 2.38** `[27.97, 34.66, 31.13, 30.57, 30.90]`, Recovery **0.992 ± 0.208**.
  - `goal-4:6` (10 Hz): **27.96 ± 1.81** `[30.41, 27.02, 29.33, 26.15, 26.92]`, Recovery **0.724 ± 0.158**.
  - `hand-8:6` (10 Hz): **23.06 ± 4.50** `[22.40, 27.33, 27.51, 16.81, 21.25]`, Recovery **0.296 ± 0.392**.
  - Welch two-sided $t$-test ($n = 5$): `hand-8` vs `goal-yaw` **$p = 0.0124$** ($n = 3$ was $0.0989$; Mann-Whitney $U = 0, p = 0.0079$, all 5 `goal-yaw` seeds above all 5 `hand-8` seeds); `hand-8` vs `goal-4` $p = 0.0704$ ($n = 3$ was $0.1972$); `goal-yaw` vs `goal-4` $p = 0.0527$. Under the pre-registered stopping rule ($p < 0.05$), **the 10 Hz reversal holds significantly on M2 for the task-native goal intent (`goal-yaw`) against `hand-8`**.

### M1 hand-8 at 10 Hz resumed 5000 → 10000 — SUBMITTED 2026-11-18 (user request)

- Why: the M1 hand-8 10 Hz collapse (5 seeds, recovery 0.09–0.30 vs 0.41–0.97 at 60 Hz, Mann-Whitney p = 0.008) was only trained to 5000 iterations, and the curves were still rising slowly. Question: does doubling training close the gap, or does it plateau well below 60 Hz?
- Job **`1162543`** (`25_train_and_eval.sbatch`, `--array=0-4`, `gpu,gpu-long`, 3 h), via `cluster_smoke/submit_m1_hand8_hold6_resume.sh`: `STUDENT_STAGE=m1`, `STUDENT_INTENTS=hand-8:6`, `STUDENT_SEEDS="0 1 2 3 4"`, `STUDENT_ITERATIONS=10000`, `STUDENT_RESUME_FROM` = newest `ep_5000` file of `1111584_6–8` (seeds 0–2) and `1123488_8–9` (seeds 3–4). Plan print checked (names `student-m1-hand-8-hold6-seed{0..4}-resume`, seed and hold match source). Cluster code md5 = local for 16/17/25, train_student, evaluate_students, intent wrapper/modules.
- Evaluation inside the job: Stage 1 task, intent hold 6, checkpoints 5200–10000.
- Acceptance: `train_and_eval_status=PASS` per task; first resumed checkpoints continue the 5000 value. Submitted only — not a result.

### M1 hand-8 10 Hz resume — RESULT (2026-11-18)

- `1162543_0–4` all COMPLETED 0:0, `train_and_eval_status=PASS`; restore continuous (5200 within 0.2 of 5000). Table: `results/m1-hand8-hold6-resume-job1162543/README.md`.
- Recovery 5000 → 10000: 0.295→0.255, 0.090→0.154, 0.149→0.200, 0.298→0.334, 0.102→0.140 (mean 0.19 → 0.22). Flat after ~8000. Max 10 Hz at 10000 (0.34) < min 60 Hz at 5000 (0.41). Answer: the collapse holds with twice the training.
