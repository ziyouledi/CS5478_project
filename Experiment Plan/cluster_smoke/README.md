# SoC Cluster Shadow Hand Smoke Test

## Goal

Produce reproducible evidence for:

1. the allocated GPU and driver;
2. the cluster OS, glibc, Python and container tooling;
3. successful creation and stepping of `Isaac-Repose-Cube-Shadow-Direct-v0`;
4. successful five-iteration RL-Games PPO run and elapsed time.

The initial SoC scripts constrain jobs to `xgpf` (Tesla T4) because a generic GPU request landed on `xgpe2`, where Slurm set `CUDA_VISIBLE_DEVICES=0` but `nvidia-smi` reported no device. A targeted `xgpf` allocation was verified to expose a working Tesla T4 with driver 580.178.04. Revisit the constraint after benchmarking other GPU families.

## Copy this directory to the cluster

Run on the main Linux development machine from the repository root. The Mac may use the same command when acting as a secondary machine:

```bash
scp -r "Experiment Plan/cluster_smoke" soc-cluster:~/cluster_smoke
```

Then log in and prepare the log directory:

```bash
ssh soc-cluster
cd ~/cluster_smoke
mkdir -p logs
```

## Stage 0: probe before choosing an Isaac Lab version

```bash
job_id=$(sbatch --parsable 00_probe_gpu.sbatch)
echo "${job_id}"
squeue -j "${job_id}"
```

After completion:

```bash
cat "logs/isaac-probe-${job_id}.out"
cat "logs/isaac-probe-${job_id}.err"
```

Do not choose the Isaac Sim/Isaac Lab pair until the driver, glibc and available Python/container runtime in this log have been checked. Isaac Sim 5.x requires Python 3.11; its pip installation also requires a sufficiently new glibc. Cluster/container installation may be preferable when the host runtime is older.

## Install the pinned stack

After creating `~/software/isaaclab-stack/env_isaaclab` with Python 3.11, install the pinned stack on a CPU compute node:

```bash
job_id=$(sbatch --parsable 03_install_isaac_stack.sbatch)
echo "${job_id}"
tail -F "logs/isaac-install-${job_id}.out" "logs/isaac-install-${job_id}.err"
```

The script pins Isaac Sim 5.1.0, PyTorch 2.7.0 with CUDA 12.8, torchvision 0.22.0, Isaac Lab v2.3.1, and the RL-Games integration. Package downloads use the compute node's local `/tmp`; the environment, exact package list, and Isaac Lab commit are retained under `~/software/isaaclab-stack`.

If an install made before the EULA and `flatdict` fixes failed, retain the partially installed environment and run:

```bash
job_id=$(sbatch --parsable 04_repair_isaac_stack.sbatch)
echo "${job_id}"
```

The repair applies the upstream `flatdict==4.0.0` fix to the v2.3.1 checkout, accepts the EULA non-interactively, resumes the editable Isaac Lab installation, restores the Isaac Sim compatibility pins, and verifies imports plus `pip check`.

For an environment where the repair finished installation but its old CPU-only import check failed on `omni`, finalize without repeating the installation:

```bash
job_id=$(sbatch --parsable 05_finalize_isaac_stack.sbatch)
echo "${job_id}"
```

This verifies package metadata and module locations without importing simulator-dependent modules before `AppLauncher` has initialized `omni`. It also records three known metadata conflicts caused by mutually exclusive pins in the Isaac Sim, Isaac Lab, WandB, and Hugging Face dependency graph; the runtime gate is the subsequent GPU `AppLauncher` smoke test.

## Stage 1: environment smoke test

After installing a pinned Isaac Lab stack, submit with absolute paths:

```bash
job_id=$(sbatch --parsable \
  --export=ALL,ISAACLAB_DIR="$HOME/software/isaaclab-stack/IsaacLab",VENV_DIR="$HOME/software/isaaclab-stack/env_isaaclab" \
  01_shadow_smoke.sbatch)
echo "${job_id}"
```

Follow the output:

```bash
tail -F "logs/shadow-smoke-${job_id}.out" "logs/shadow-smoke-${job_id}.err"
```

The script treats an intentional ten-minute timeout as success if the environment kept running. Confirm that the output contains successful simulator startup, task creation, repeated stepping, and `smoke_status=PASS`.

## Stage 2: five-iteration PPO timing run

```bash
job_id=$(sbatch --parsable \
  --export=ALL,ISAACLAB_DIR="$HOME/software/isaaclab-stack/IsaacLab",VENV_DIR="$HOME/software/isaaclab-stack/env_isaaclab" \
  02_short_train.sbatch)
echo "${job_id}"
```

After completion, retain the Slurm log and the generated `logs/rl_games/...` run directory. The Slurm output records the GPU model, driver, repository commit and wall-clock time.

## Stage 3: inspect the Shadow Hand interface

```bash
job_id=$(sbatch --parsable \
  --export=ALL,ISAACLAB_DIR="$HOME/software/isaaclab-stack/IsaacLab",VENV_DIR="$HOME/software/isaaclab-stack/env_isaaclab" \
  06_inspect_shadow_interface.sbatch)
echo "${job_id}"
```

This records the runtime action-to-joint mapping, wrist indices, control timing, observation shape, goal/object tensor fields, and the exact source methods that assemble observations and apply actions.

## Stage 4: expert training-speed benchmark

Run 20 iterations with the unmodified PPO minibatch size. With 2048 environments and horizon 16, each rollout contains 32768 transitions, exactly one default minibatch:

```bash
job_id=$(sbatch --parsable \
  --export=ALL,ISAACLAB_DIR="$HOME/software/isaaclab-stack/IsaacLab",VENV_DIR="$HOME/software/isaaclab-stack/env_isaaclab" \
  07_expert_benchmark.sbatch)
echo "${job_id}"
```

The output reports total frames, elapsed time, steady-state FPS, peak GPU memory, and the last checkpoint. Use this result to choose the environment count and wall time for the `gpu-long` expert job.

## Stage 5: train the seed-0 expert

The full job reads `max_epochs`, `horizon_length`, minibatch size, and checkpoint frequency from the pinned task YAML. It uses 2048 environments, writes monitoring data under `~/data/low-dim-intent`, and can resume from a checkpoint:

```bash
job_id=$(sbatch --parsable \
  --export=ALL,ISAACLAB_DIR="$HOME/software/isaaclab-stack/IsaacLab",VENV_DIR="$HOME/software/isaaclab-stack/env_isaaclab" \
  08_train_expert.sbatch)
echo "${job_id}"
```

To resume, submit the same script with an absolute checkpoint path:

```bash
job_id=$(sbatch --parsable \
  --export=ALL,ISAACLAB_DIR="$HOME/software/isaaclab-stack/IsaacLab",VENV_DIR="$HOME/software/isaaclab-stack/env_isaaclab",CHECKPOINT="/absolute/path/to/checkpoint.pth" \
  08_train_expert.sbatch)
```

## Stage 6: compare the best and final expert checkpoints

After copying the two checkpoints to
`~/data/low-dim-intent/checkpoints/expert-seed0/`, evaluate both with the
same deterministic policy, seed, number of environments, and episode count:

```bash
job_id=$(sbatch --parsable \
  --export=ALL,ISAACLAB_DIR="$HOME/software/isaaclab-stack/IsaacLab",VENV_DIR="$HOME/software/isaaclab-stack/env_isaaclab" \
  09_evaluate_expert.sbatch)
echo "${job_id}"
```

The default checkpoint names are `expert-seed0-best.pth` and
`expert-seed0-final.pth`. The job evaluates 1024 episodes per checkpoint and
writes the exact metrics to
`~/data/low-dim-intent/runs/expert-eval-job<job-id>/{best,final}.json`.
Choose the PCA rollout checkpoint using mean episode reward first and
`mean_consecutive_successes` as the task-specific supporting metric.

## Stage 7: freeze the selected seed-0 expert

After the full checkpoint comparison, create a canonical read-only checkpoint,
checksum, evaluation copy, and manifest. Pass the completed evaluation job ID:

```bash
EVAL_JOB_ID=JOB_ID bash 10_freeze_expert.sh final
```

The canonical checkpoint is
`~/data/low-dim-intent/checkpoints/expert-seed0/frozen/expert-seed0.pth`.
All rollout and student-policy jobs must use this path. Do not select a source
checkpoint by timestamp or by scanning the RL-Games log directory.

## Stage 8: collect a deterministic expert rollout

The default pilot uses 256 environments for 700 vector steps, producing
179,200 transitions and crossing the task's 600-step time limit once:

```bash
job_id=$(sbatch --parsable \
  --export=ALL,ISAACLAB_DIR="$HOME/software/isaaclab-stack/IsaacLab",VENV_DIR="$HOME/software/isaaclab-stack/env_isaaclab" \
  11_collect_expert_rollout.sbatch)
echo "${job_id}"
```

The dataset contains memory-mapped NumPy arrays for the 157-D observation,
20-D expert action, reward, done flag, environment ID, episode ID, and episode
step. `manifest.json` records the checkpoint hash, exact shapes and dtypes,
observation slices, runtime action order, summary metrics, and collection rate.

After validating the pilot, collect the PCA dataset with 512 environments and
2400 vector steps (1,228,800 transitions):

```bash
job_id=$(sbatch --parsable \
  --export=ALL,ISAACLAB_DIR="$HOME/software/isaaclab-stack/IsaacLab",VENV_DIR="$HOME/software/isaaclab-stack/env_isaaclab",ROLLOUT_NUM_ENVS=512,ROLLOUT_VECTOR_STEPS=2400 \
  11_collect_expert_rollout.sbatch)
echo "${job_id}"
```

## Stage 9: action PCA basis

This is a CPU-only job (`normal` partition, no Isaac Sim). It first runs
`sha256sum -c SHA256SUMS` inside the rollout directory, then computes the 20-D
robot basis (`mu_robot`, `W_robot`) and the separate 18-D finger basis
(`mu_finger`, `W_finger`, action columns 2:20) for `Wrist-2 + Finger-k`:

```bash
job_id=$(sbatch --parsable \
  --export=ALL,VENV_DIR="$HOME/software/isaaclab-stack/env_isaaclab",PCA_DATASET_DIR="$HOME/data/low-dim-intent/datasets/expert-rollout-seed0-job1023537" \
  12_action_pca.sbatch)
echo "${job_id}"
```

Output goes to `~/data/low-dim-intent/datasets/action-pca-<rollout>-job<JOB_ID>/`:
`pca_basis.npz`, `summary.json` (provenance, explained variance, components
needed for 80/90/95/99%, reconstruction checks), `cumulative_variance.csv`,
`cumulative_variance.png`, and `SHA256SUMS`. Columns of `W` are components in
descending-variance order; intent is `z_k = W[:, :k].T @ (a - mu)`. Each
component's largest-magnitude loading is positive. PCA uses the stored
(unclipped) expert actions unless `PCA_CLIP_ACTIONS` is set; `summary.json`
reports the fraction of entries outside [-1, 1]. Success requires
`dataset_checksum_verification=PASS`, `pca_status=PASS`, and `pca_job_status=PASS`.

## Useful monitoring commands

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,JobName,Partition,State,Elapsed,AllocTRES,NodeList
scancel JOB_ID
```
