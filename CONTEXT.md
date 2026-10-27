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
- **PCA job submitted**: Slurm `1024096` (CPU, `normal`, `12_action_pca.sbatch` on dataset `1023537`, unclipped actions). Not yet a result: check `sacct -j 1024096`, `dataset_checksum_verification=PASS`, `pca_status=PASS`, `pca_job_status=PASS`, and `datasets/action-pca-expert-rollout-seed0-job1023537-job1024096/`. Review `fraction_entries_outside_unit_range` before deciding whether clipping matters.
- **Next gate — PCA basis**: from the formal `actions.npy`, compute the 20-D action mean `μ`, principal components `W_robot`, explained variance, and cumulative variance plot, plus a separate 18-D finger basis for `Wrist-2 + Finger-k`. Record input dataset/checkpoint hashes and action-column order.
- Then implement and dry-run Hand-k, Wrist-2, and Obj-1/4 intent inputs before launching matched student-policy runs. No modules or student results exist yet.
- Expert evaluation runs exist under `runs/expert-eval-job{915339,915716,915789,915824}`; which one is the formal 1024-episode evaluation has not been recorded.

The collector writes NumPy arrays for pre-action observation, deterministic expert action, resulting reward/done, environment ID, episode ID, and episode step. Its `manifest.json` includes shapes, dtypes, provenance, and summary metrics. It writes a `.partial` directory first and renames it on completion. RL-Games batch playback requires `player.has_batch_dimension = True`; the current collector and evaluator include this fix.

## Where to work

- Local source of truth: this Mac repository. Slurm scripts and Python helpers are in [`Experiment Plan/cluster_smoke/`](Experiment%20Plan/cluster_smoke/); its `README.md` has concrete submission commands.
- Cluster persistent software: `~/software/isaaclab-stack/`; experiment outputs: `~/data/low-dim-intent/{runs,checkpoints,datasets,cache,metadata}/`. Job-local `/tmp` is temporary.
- Git tracks code, configs, and small reproducibility summaries. Large arrays, checkpoints, logs, and caches stay in cluster storage. At the time of this snapshot, several local experiment scripts and the plan had uncommitted changes; inspect `git status` before syncing or committing.
- The detailed design document still contains its original roadmap dates; a note at its top points to the current schedule (demo 11-18, report end of November). Re-plan the roadmap gates against those dates.
