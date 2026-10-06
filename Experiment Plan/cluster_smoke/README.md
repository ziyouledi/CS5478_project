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

## Stage 1: environment smoke test

After installing a pinned Isaac Lab stack, submit with absolute paths:

```bash
job_id=$(sbatch --parsable \
  --export=ALL,ISAACLAB_DIR="$HOME/isaaclab-stack/IsaacLab",VENV_DIR="$HOME/isaaclab-stack/env_isaaclab" \
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
  --export=ALL,ISAACLAB_DIR="$HOME/isaaclab-stack/IsaacLab",VENV_DIR="$HOME/isaaclab-stack/env_isaaclab" \
  02_short_train.sbatch)
echo "${job_id}"
```

After completion, retain the Slurm log and the generated `logs/rl_games/...` run directory. The Slurm output records the GPU model, driver, repository commit and wall-clock time.

## Useful monitoring commands

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,JobName,Partition,State,Elapsed,AllocTRES,NodeList
scancel JOB_ID
```
