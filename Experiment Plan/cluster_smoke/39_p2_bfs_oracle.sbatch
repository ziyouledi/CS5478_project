#!/usr/bin/env bash
#SBATCH --job-name=p2-bfs-oracle
#SBATCH --partition=gpu,gpu-long
#SBATCH --constraint=xgpf
#SBATCH --gpus=1
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Priority 2.1 (A0 gate): external-goal wrapper, hold stability (60/120/240 steps),
# and Closed-loop vs Open-loop 96-state BFS Oracle on frozen Unified v4.

set -Eeuo pipefail

ISAACLAB_DIR="${ISAACLAB_DIR:-${HOME}/software/isaaclab-stack/IsaacLab}"
VENV_DIR="${VENV_DIR:-${HOME}/software/isaaclab-stack/env_isaaclab}"
CHECKPOINT="${P2_CHECKPOINT:-${HOME}/data/low-dim-intent/checkpoints/expert-unified-cap0.05_hold10_v4-seed0/frozen/expert-unified-v4-seed0.pth}"
NUM_ENVS="${P2_NUM_ENVS:-256}"
SEED="${P2_SEED:-20261120}"
TAG="${P2_TAG:-v4-seed0}"
OUT_DIR="${HOME}/data/low-dim-intent/runs/p2-bfs-oracle-${TAG}-job${SLURM_JOB_ID}"

mkdir -p "${OUT_DIR}" logs
cp "${SLURM_SUBMIT_DIR}"/{cube_state_graph.py,p2_bfs_oracle.py,39_p2_bfs_oracle.sbatch} "${OUT_DIR}/"

# Unified v4 env settings (must match training/eval of expert-unified-v4-seed0)
export OMNI_KIT_ACCEPT_EULA=YES
export XDG_CACHE_HOME="${HOME}/data/low-dim-intent/cache/xdg"
export HINGE_EFFORT_LIMIT=0.05
export FACE_TURN_HOLD_STEPS=10
export UNIFIED_TURN_ORIENT_TOL=0.4
export UNIFIED_EXTRA_OBS=1
export UNIFIED_MAX_PHASE_STEPS=0
export UNIFIED_TIMEOUT_TERMINATES=0
export HINGE_ASSET_DIR="${OUT_DIR}/asset"
export HINGE_REFERENCE_JSON="${HOME}/data/low-dim-intent/runs/hinged-cube-eval-job1141499/dexcube.json"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

source "${VENV_DIR}/bin/activate"
cd "${ISAACLAB_DIR}"

echo "record_type=p2_bfs_oracle_job job_id=${SLURM_JOB_ID} node=$(hostname -f) timestamp=$(date --iso-8601=seconds)"
echo "gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | paste -sd ';')"
echo "checkpoint=${CHECKPOINT} num_envs=${NUM_ENVS} seed=${SEED} out_dir=${OUT_DIR}"

"${VENV_DIR}/bin/python" "${SLURM_SUBMIT_DIR}/p2_bfs_oracle.py" \
    --headless \
    --checkpoint "${CHECKPOINT}" \
    --num_envs "${NUM_ENVS}" \
    --seed "${SEED}" \
    --output_dir "${OUT_DIR}" \
    2>&1 | tee "${OUT_DIR}/eval.log"

grep -q "p2_bfs_oracle_status=PASS" "${OUT_DIR}/eval.log"
echo "p2_bfs_oracle_job_status=PASS"
