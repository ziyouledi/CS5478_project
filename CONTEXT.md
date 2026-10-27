# Project context — Shadow Hand intent experiments

Last reconciled: 2026-10-27 (cluster-verified), Asia/Singapore. This is the working snapshot for agents. Read [HANDOFF.md](HANDOFF.md) for detailed history, job results, data layout, and caveats; read [the v3 experiment plan](Docs/Experiment%20Plan/低维意图接口实验设计%20v3（分阶段）.md) for hypotheses and comparison design. Current schedule (user, 2026-10-27): **Project Milestone due 2026-10-27** (not yet submitted at reconciliation time), **demo 2026-11-18**, **final report end of November 2026**. These supersede the plan's older dates and the earlier 10-20 / 11-22 dates.

## Objective

Use Isaac Lab's `Isaac-Repose-Cube-Shadow-Direct-v0` to compare what information a low-dimensional intent interface should send to a Shadow Hand controller. Stage 1 keeps the goal visible and compares Blind, Wrist-2, Hand-2/4/20, and Obj-1/4 for learning efficiency. Stage 2 M1 hides the goal from the actor and measures performance recovery from intent. Planned comparison seeds are `0, 1, 2` with matched training and evaluation budgets. M2 (another manipulation task) and M3 (human synergy basis) stay in scope for now (user decision 2026-10-27, not cut); final choice later depending on progress.

## Confirmed baseline

| Item | Evidence |
| --- | --- |
| Runtime | Isaac Lab v2.3.1 at `5c2ec81cb17532d32f7922dd7fcaae40d123b71a`; Isaac Sim 5.1.0.0; Python 3.11.17; PyTorch 2.7.0+cu128; RL-Games 1.6.1 |
| GPU | Tesla T4, 15,360 MiB, driver 580.178.04; existing GPU scripts constrain Slurm to `xgpf` |
| Task interface | 20-D action, 157-D policy observation, 120 Hz physics, 60 Hz control, episode limit about 600 control steps |
| Action map | Indices `0:2` are the two wrist joints; `2:20` are 18 finger drives. Use runtime `actuated_dof_indices`, not the config list order. Exact mapping is in `collect_shadow_rollout.py`. |
| Observation | Exact 157-D field slices are in `collect_shadow_rollout.py`. Slice `[137:157]` is the previous action. |

Seed-0 expert training completed as Slurm job `915072`: 2048 environments, 5000 iterations, 163,840,000 transitions, 1:34:11 elapsed. The independent deterministic evaluation used 1024 episodes per checkpoint:

| Checkpoint | Mean episode reward | Reward SD | Mean consecutive successes |
| --- | ---: | ---: | ---: |
| best | 7497.23 | 1052.85 | 14.74 |
| final | 7447.82 | 945.64 | 15.49 |

The experiment selected **final** for rollout because reward differed by about 0.66%, while final had about 5.07% higher consecutive successes and lower reward variance. This is a practical selection, not a reported significance test. `consecutive_successes` comes from `infos.episode.consecutive_successes`; it is not a binary episode success rate.

The intended frozen expert path is `~/data/low-dim-intent/checkpoints/expert-seed0/frozen/expert-seed0.pth`. `10_freeze_expert.sh` creates a read-only copy, SHA-256 file, evaluation copy, and manifest. This conversation does not contain the actual freeze command output, so verify its manifest/checksum before reusing it.

## Current status and next gate

Verified on the cluster on 2026-10-27 (login-node inspection of `sacct`, logs, and manifests):

- **Frozen expert**: `~/data/low-dim-intent/checkpoints/expert-seed0/frozen/` exists (created 2026-10-06, read-only) with `expert-seed0.pth`, `SHA256SUMS`, `evaluation.json`, `manifest.json`. SHA-256 `b9b7128f151b44c8241603e0138d5110435e70e13fad902431a0bbe96f2074f9`.
- **Pilot rollout**: jobs `1023495` and `1023529`, both `COMPLETED 0:0`, `256 × 700 = 179,200` transitions each, identical summary metrics (deterministic collection reproduces). Pipeline validation only.
- **Formal PCA rollout: DONE.** Job `1023537`, `COMPLETED 0:0`, log shows `rollout_samples=1228800`, `rollout_collection_status=PASS`, `rollout_job_status=PASS`. Dataset `~/data/low-dim-intent/datasets/expert-rollout-seed0-job1023537/` (806 MB, `512 × 2400`, collection seed `20261007`, checkpoint hash matches the frozen expert). Manifest summary: 2050 completed episodes, mean episode reward 7486.40, mean episode length 592.7. `sha256sum -c SHA256SUMS` has not yet been run on it; do so inside the PCA Slurm job rather than on the login node.
- **Metric caveat**: pilot mean consecutive successes is 3.38 vs 16.25 for the formal run at near-identical episode reward. The Isaac Lab metric appears to be a running average that warms up over resets, so short runs read low. Confirm its definition before using it as a student-policy metric.
- **PCA basis: DONE.** Slurm `1024096` (`COMPLETED 0:0`, 44 s), dataset checksums verified in-job (`dataset_checksum_verification=PASS`), `pca_status=PASS`, `pca_job_status=PASS`. Output `~/data/low-dim-intent/datasets/action-pca-expert-rollout-seed0-job1023537-job1024096/` (`pca_basis.npz` with `mu_robot`, `W_robot`, `mu_finger`, `W_finger`). Summary, CSV, plot, and checksums copied to `Experiment Plan/results/action-pca-job1024096/`.
  - Stored expert actions already lie in [-1, 1] (0 entries outside), so the clipping question is moot.
  - Spectrum is flat: 20-D cumulative variance k=2 35.8%, k=4 56.8%, k=8 81.1%; 95% needs 14 components. 18-D finger: k=2 40.8%, k=4 59.9%, k=8 83.1%; 95% needs 13.
  - Two near-constant actions (`robot0_MFJ2` std 0.003, max −0.277; `robot0_RFJ3` std 0.028) explain the ~0 tail eigenvalues. Wrist loads mainly on PC3 (59% of its squared loading).
- **Action saturation: measured.** Slurm `1024114` (`COMPLETED 0:0`, checksums PASS, `saturation_job_status=PASS`); summary and plot in `Experiment Plan/results/action-saturation-job1024114/`.
  - 62.2% of all action entries are at a bound (|a| ≥ 0.99); on average 12.4 of 20 dims are saturated per step.
  - The expert is near bang-bang with chatter: most active dims flip sign between consecutive control steps 20–32% of the time, mean |Δa| per step ≈ 0.4–0.6. Direct −1↔+1 jumps (both steps saturated) occur 2–18% of the time depending on the joint.
  - `act_moving_average=1.0`, so these raw actions are the joint position targets with no env-side smoothing.
  - `MFJ2` is pinned at −1 100% of the time, `RFJ3` 99%; `MFJ3`, `LFJ3`, `THJ2` are > 80% at −1.
  - Implication for Hand-k: linear PCA on these targets mixes bounded switching with chatter; low-k intents capture limited variance (see PCA bullet). Any decision to change the intent source (e.g. smoothed targets or joint positions) is a scientific change that needs user approval.
- **Intent modules: implemented and validated.** `intent_modules.py` + `train_student.py` (Stages 11–12 in the cluster README).
  - CPU validation `1024158` (`COMPLETED 0:0`, `intent_validation_status=PASS`, 131,072 rollout rows): rebuilt `FrozenExpert` matches recorded player actions within 5.3e-6; Hand-20 reconstructs the action within 7.7e-7; latent variances within 0.4% of PCA eigenvalues; object relative quaternion matches the env's `obs[68:72]` within 2.4e-7. The first attempt `1024139` failed only because the Obj-1 angle tolerance (1e-4) was tighter than float32 asin precision near pi; tolerance set to 2e-3.
  - GPU dry run `1024366` (`COMPLETED 0:0`, `student_dry_run_job_status=PASS`, 256 envs × 3 iterations): blind 157-D, wrist-2/hand-2 159, hand-4/obj-4 161, hand-20 177, obj-1 158; tensorboard summaries and `params/intent.json` written. The earlier attempt `1024140` was cancelled: legacy `gym.spaces.Box` vs gymnasium type check failed at start, and Isaac Sim hung on `close()` until each 900 s timeout. Both fixed (keep the base Box class; `os._exit(1)` on failure; dry run stops at first failure).
- **Decided (user, 2026-10-27)**: Obj-3/4 lookahead Δθ = **0.5 rad**; stage 1 keeps the **shared actor-critic** (`separate: False`), so the critic also sees the intent — state this in the report.
- **Decided (user, 2026-10-27)**: 5000 PPO iterations per student run (same as expert, 2048 envs, 163.84M transitions); at most **4 concurrent GPUs**; final evaluation uses **per-episode success counts + episode reward**, not the EMA `consecutive_successes`. Learning curves / N80 will come from offline evaluation of the checkpoints saved every 200 iterations.
- **Stage-1 student grid submitted**: Slurm array `1024454` (`--array=0-20%4`), `16_train_student.sbatch`, intents `blind wrist-2 hand-2 hand-4 hand-20 obj-1 obj-4` × seeds `0 1 2` (task i → intent i//3, seed i%3), Δθ 0.5. Partition widened to `gpu,gpu-long` with 3 h limit after all xgpf GPUs were busy. Outputs: `runs/student-stage1-<intent>-seed<s>-job1024454_<i>/`. Submitted only — not a result. Check `sacct -j 1024454` and `student_job_status=PASS` per task.
- **Next**: write the per-episode evaluation script for student checkpoints (deterministic, fixed seed, 1024 episodes, success count per episode from the success-bonus events or `rot_dist <= success_tolerance`), then compute N80 / AUC. `consecutive_successes` is confirmed to be an EMA over resets (`av_factor`) in `inhand_manipulation_env.py`, so per-episode success counts should be used for final evaluation.
- Expert evaluation runs exist under `runs/expert-eval-job{915339,915716,915789,915824}`; which one is the formal 1024-episode evaluation has not been recorded.

The collector writes NumPy arrays for pre-action observation, deterministic expert action, resulting reward/done, environment ID, episode ID, and episode step. Its `manifest.json` includes shapes, dtypes, provenance, and summary metrics. It writes a `.partial` directory first and renames it on completion. RL-Games batch playback requires `player.has_batch_dimension = True`; the current collector and evaluator include this fix.

## Where to work

- Local source of truth: this Mac repository. Slurm scripts and Python helpers are in [`Experiment Plan/cluster_smoke/`](Experiment%20Plan/cluster_smoke/); its `README.md` has concrete submission commands.
- Cluster persistent software: `~/software/isaaclab-stack/`; experiment outputs: `~/data/low-dim-intent/{runs,checkpoints,datasets,cache,metadata}/`. Job-local `/tmp` is temporary.
- Git tracks code, configs, and small reproducibility summaries. Large arrays, checkpoints, logs, and caches stay in cluster storage. At the time of this snapshot, several local experiment scripts and the plan had uncommitted changes; inspect `git status` before syncing or committing.
- The detailed design document still contains its original roadmap dates; a note at its top points to the current schedule (demo 11-18, report end of November). Re-plan the roadmap gates against those dates.
