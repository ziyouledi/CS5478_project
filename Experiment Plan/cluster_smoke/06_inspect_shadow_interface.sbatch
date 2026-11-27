#!/usr/bin/env bash
#SBATCH --job-name=shadow-interface
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=xgpf
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -Eeuo pipefail

: "${ISAACLAB_DIR:?Submit with ISAACLAB_DIR and VENV_DIR}"
: "${VENV_DIR:?Submit with ISAACLAB_DIR and VENV_DIR}"

source "${VENV_DIR}/bin/activate"
cd "${ISAACLAB_DIR}"

export PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA=YES
export XDG_CACHE_HOME="${HOME}/data/low-dim-intent/cache/xdg"
mkdir -p "${XDG_CACHE_HOME}"

echo "record_type=shadow_interface_inspection"
echo "timestamp=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname -f)"
echo "isaaclab_commit=$(git rev-parse HEAD)"
echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd ';')"

/usr/bin/time -p timeout --signal=INT --kill-after=30s 720s \
    ./isaaclab.sh -p "${SLURM_SUBMIT_DIR}/inspect_shadow_interface.py" \
    --task Isaac-Repose-Cube-Shadow-Direct-v0 \
    --num_envs 1 \
    --headless

echo "inspection_status=PASS"
