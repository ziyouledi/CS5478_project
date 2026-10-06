#!/usr/bin/env bash
#SBATCH --job-name=shadow-smoke
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

echo "record_type=shadow_random_agent_smoke"
echo "timestamp=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname -f)"
echo "partition=${SLURM_JOB_PARTITION}"
echo "isaaclab_dir=${ISAACLAB_DIR}"
echo "isaaclab_commit=$(git rev-parse HEAD)"
echo "python=$(python --version 2>&1)"
echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd ';')"
echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | paste -sd ';')"

start_epoch=$(date +%s)
set +e
/usr/bin/time -p timeout --signal=INT --kill-after=30s 600s \
    ./isaaclab.sh -p scripts/environments/random_agent.py \
    --task Isaac-Repose-Cube-Shadow-Direct-v0 \
    --num_envs 64 \
    --headless
command_status=$?
set -e
end_epoch=$(date +%s)

echo "elapsed_seconds=$((end_epoch - start_epoch))"
echo "command_status=${command_status}"

# timeout returns 124 when the environment ran until the planned cutoff.
if [[ "${command_status}" -eq 0 || "${command_status}" -eq 124 || "${command_status}" -eq 130 ]]; then
    echo "smoke_status=PASS"
else
    echo "smoke_status=FAIL"
    exit "${command_status}"
fi
