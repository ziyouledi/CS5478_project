#!/usr/bin/env bash
#SBATCH --job-name=vlm-demo-t4
#SBATCH --partition=gpu,gpu-long
#SBATCH --gpus=1
#SBATCH --constraint=xgpf
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Records 60 fps end-to-end VLM + Unified v4 closed-loop demo videos (num_envs=1)
# connected to the H100 vLLM server (45_vllm_server_for_isaac.sbatch).

set -Eeuo pipefail

ISAAC_VENV="${HOME}/software/isaaclab-stack/env_isaaclab"
ISAACLAB_DIR="${HOME}/software/isaaclab-stack/IsaacLab"
V4_CKPT="${HOME}/data/low-dim-intent/checkpoints/expert-unified-cap0.05_hold10_v4-seed0/frozen/expert-unified-v4-seed0.pth"
COORD_DIR="${CL_COORD_DIR:-${HOME}/data/low-dim-intent/runs/vlm-demo-coord}"
TAG="${CL_TAG:-c1-v2-demo}"

ENDPOINT_FILE="${COORD_DIR}/endpoint.txt"
DONE_FLAG="${COORD_DIR}/done.flag"
OUT_DIR="${HOME}/data/low-dim-intent/videos/vlm-closedloop-${TAG}-job${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}" "${COORD_DIR}" logs
cp "${SLURM_SUBMIT_DIR}"/{record_vlm_closedloop_demo.py,48_record_vlm_demo_t4.sbatch} "${OUT_DIR}/"

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
# Exact 3 VLM cameras (c1, s040, s220) + 1 wider hero spectator camera (demo)
export Z1_CAMERAS="c1:-50:38:0.33,s040:40:30:0.33,s220:-140:30:0.33,demo:-50:42:0.55"
export Z1_IMAGE="512" Z1_FOCAL="24" Z1_TARGET="0 -0.39 0.6"
mkdir -p "${XDG_CACHE_HOME}" "${HINGE_ASSET_DIR}"
unset UNIFIED_CYCLE UNIFIED_START_STATES FACE_TURN_START_STATES HINGE_STIFFNESS HINGE_DAMPING

echo "record_type=vlm_demo_t4 job_id=${SLURM_JOB_ID} node=$(hostname -f) tag=${TAG} out_dir=${OUT_DIR}"
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
    "${ISAAC_VENV}/bin/python" "${SLURM_SUBMIT_DIR}/record_vlm_closedloop_demo.py" \
        --headless \
        --checkpoint "${V4_CKPT}" \
        --vllm_url "${ENDPOINT}/v1/chat/completions" \
        --vllm_model "vlm_planner" \
        --max_round_steps 120 \
        --output_dir "${OUT_DIR}" \
        2>&1 | tee "${OUT_DIR}/record_demo.log"
)
grep -q "record_vlm_closedloop_demo_status=PASS" "${OUT_DIR}/record_demo.log"
echo "vlm_demo_t4_status=PASS"
