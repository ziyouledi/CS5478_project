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
- [ ] 1. **Evaluation-only hold (GPU, no training).** Existing M1 final checkpoints (`ep_5000`) of hand-2, hand-8, obj-4 × seeds 0–2 from `1104420`, evaluated with N = 1, 6, 30, 120.
  - Code: hold option in `intent_env_wrapper.py` (default N = 1, behaviour unchanged) and an `EVAL_INTENT_HOLD` override in `evaluate_students.py` / `17`.
  - Check: N = 1 reproduces the recorded M1 final value within the evaluator's order effect (~0.5%).
  - Caveat: these students were trained at 60 Hz, so the result is a lower bound for step 2.
- [ ] 2. **Training grid.**
  - Code: `goal-4` in `intent_modules.py`; `--intent_hold` in `train_student.py`, written to `intent.json` and read by the evaluator; `STUDENT_INTENT_HOLD` in `15`/`16`. CPU test of the hold logic, then a GPU dry run (256 envs × 3 iterations) for every (intent, N) pair.
  - Conditions: hand-2, hand-8, obj-4, goal-4 × N = 6, 30, 120 × 3 seeds = 36 runs, plus goal-4 at N = 1 × 3 seeds = **39 runs**. hand-2, hand-8, obj-4 at N = 1 are the existing M1 runs.
  - Cost: an M1 run takes about 1:47. With 3 training slots and 1 evaluation slot, 39 runs take about 23 hours of wall time. Evaluations (12 min each) fit in the fourth slot. Submit in batches under the 32-job limit, with `aftercorr` evaluations submitted while training is still pending.
  - Pruning rule: if steps 0 and 1 show that hand-k keeps its information at N = 30 and 120 (the prediction is wrong), drop hand-2 and keep hand-8, obj-4, goal-4 (30 runs).
- [ ] 3. **Analysis.** Recovery, AUC and N80 per (intent, N); Recovery against update rate, one line per intent; Welch tests against the same intent at N = 1. Copy summaries to `results/rate-sweep-*/`, update `CONTEXT.md`.

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

