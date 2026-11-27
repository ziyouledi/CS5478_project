#!/usr/bin/env bash
#SBATCH --job-name=isaac-cl-t4
#SBATCH --partition=gpu,gpu-long
#SBATCH --gpus=1
#SBATCH --constraint=xgpf
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# T4 (xgpf, RT-core verified) Isaac Lab 60 Hz Unified v4 + live 3-camera rendering client.
# Connects over cluster LAN to the H100 vLLM server (45_vllm_server_for_isaac.sbatch).

set -Eeuo pipefail

ISAAC_VENV="${HOME}/software/isaaclab-stack/env_isaaclab"
ISAACLAB_DIR="${HOME}/software/isaaclab-stack/IsaacLab"
V4_CKPT="${HOME}/data/low-dim-intent/checkpoints/expert-unified-cap0.05_hold10_v4-seed0/frozen/expert-unified-v4-seed0.pth"
COORD_DIR="${CL_COORD_DIR:-${HOME}/data/low-dim-intent/runs/isaac-cl-coord}"
TAG="${CL_TAG:-c1-vision-full}"

ENDPOINT_FILE="${COORD_DIR}/endpoint.txt"
DONE_FLAG="${COORD_DIR}/done.flag"
OUT_DIR="${HOME}/data/low-dim-intent/runs/isaac-closedloop-${TAG}-job${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}" "${COORD_DIR}" logs
cp "${SLURM_SUBMIT_DIR}"/{eval_c2_isaac_closedloop.py,46_isaac_closedloop_t4.sbatch} "${OUT_DIR}/"

signal_done() {
    touch "${DONE_FLAG}"
}
trap signal_done EXIT

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"
export OMNI_KIT_ACCEPT_EULA=YES
export XDG_CACHE_HOME="${HOME}/data/low-dim-intent/cache/xdg"
export HINGE_ASSET_DIR="${OUT_DIR}/asset"
export HINGE_EFFORT_LIMIT=0.05
export FACE_TURN_HOLD_STEPS=10
export HINGE_REFERENCE_JSON="${HOME}/data/low-dim-intent/runs/hinged-cube-eval-job1141499/dexcube.json"
export UNIFIED_EXTRA_OBS=1
export Z1_CAMERAS="c1:-50:38:0.33,s040:40:30:0.33,s220:-140:30:0.33"
export Z1_IMAGE="512" Z1_FOCAL="24" Z1_TARGET="0 -0.39 0.6"
mkdir -p "${XDG_CACHE_HOME}" "${HINGE_ASSET_DIR}"
unset UNIFIED_CYCLE UNIFIED_START_STATES FACE_TURN_START_STATES HINGE_STIFFNESS HINGE_DAMPING

echo "record_type=isaac_cl_t4 job_id=${SLURM_JOB_ID} node=$(hostname -f) tag=${TAG} out_dir=${OUT_DIR}"
echo "gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | paste -sd ';')"

echo "Waiting for vLLM server endpoint at ${ENDPOINT_FILE}..."
ENDPOINT=""
for i in $(seq 1 300); do
    if [[ -s "${ENDPOINT_FILE}" ]]; then
        CAND="$(cat "${ENDPOINT_FILE}" | tr -d '[:space:]')"
        if curl -sf "${CAND}/health" >/dev/null 2>&1; then
            ENDPOINT="${CAND}"
            echo "Connected to vLLM server at ${ENDPOINT} after $((i * 2))s"
            break
        fi
    fi
    sleep 2
done
if [[ -z "${ENDPOINT}" ]]; then
    echo "ERROR: Timed out waiting for vLLM server at ${ENDPOINT_FILE}"
    exit 1
fi

(
    cd "${ISAACLAB_DIR}"
    "${ISAAC_VENV}/bin/python" "${SLURM_SUBMIT_DIR}/eval_c2_isaac_closedloop.py" \
        --headless \
        --checkpoint "${V4_CKPT}" \
        --vllm_url "${ENDPOINT}/v1/chat/completions" \
        --vllm_model "vlm_planner" \
        --num_envs 8 \
        --batches_per_level 4 \
        --levels "1,2,3,4" \
        --max_round_steps 120 \
        --output_dir "${OUT_DIR}" \
        2>&1 | tee "${OUT_DIR}/isaac_128.log"
)
grep -q "eval_c2_isaac_closedloop_status=PASS" "${OUT_DIR}/isaac_128.log"
echo "isaac_cl_t4_status=PASS"
