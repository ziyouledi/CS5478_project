#!/usr/bin/env bash
#SBATCH --job-name=shadow-ppo-short
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=xgpf
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -Eeuo pipefail

: "${ISAACLAB_DIR:?Submit with --export=ALL,ISAACLAB_DIR=/absolute/path/to/IsaacLab,VENV_DIR=/absolute/path/to/env_isaaclab}"
: "${VENV_DIR:?Submit with --export=ALL,ISAACLAB_DIR=/absolute/path/to/IsaacLab,VENV_DIR=/absolute/path/to/env_isaaclab}"

source "${VENV_DIR}/bin/activate"
cd "${ISAACLAB_DIR}"

export PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA=YES

echo "record_type=shadow_ppo_short"
echo "timestamp=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname -f)"
echo "partition=${SLURM_JOB_PARTITION}"
echo "isaaclab_commit=$(git rev-parse HEAD)"
echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd ';')"
echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | paste -sd ';')"

start_epoch=$(date +%s)
/usr/bin/time -p ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Repose-Cube-Shadow-Direct-v0 \
    --num_envs 512 \
    --max_iterations 5 \
    --seed 0 \
    --headless
end_epoch=$(date +%s)

echo "elapsed_seconds=$((end_epoch - start_epoch))"
echo "short_train_status=PASS"
