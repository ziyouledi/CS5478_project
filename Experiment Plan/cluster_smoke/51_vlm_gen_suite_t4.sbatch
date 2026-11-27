#!/usr/bin/env bash
#SBATCH --job-name=vlm-gen-suite
#SBATCH --partition=gpu,gpu-long
#SBATCH --gpus=1
#SBATCH --constraint=xgpf
#SBATCH --time=01:15:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Plan B Generalization Suite on T4 connected to H100 vLLM server (45_vllm_server_for_isaac.sbatch):
#   Part 1: Record 3 hardcore closed-loop demo videos (dark low-light 200, OOD green/blue swap, pure casual spoken prompt)
#   Part 2a: 32-episode quantitative closed-loop eval under dark low-light (intensity=200, L=1..4)
#   Part 2b: 32-episode quantitative closed-loop eval under pure casual spoken prompts (zero schema hints, L=1..4)

set -Eeuo pipefail

ISAAC_VENV="${HOME}/software/isaaclab-stack/env_isaaclab"
ISAACLAB_DIR="${HOME}/software/isaaclab-stack/IsaacLab"
V4_CKPT="${HOME}/data/low-dim-intent/checkpoints/expert-unified-cap0.05_hold10_v4-seed0/frozen/expert-unified-v4-seed0.pth"
COORD_DIR="${CL_COORD_DIR:-${HOME}/data/low-dim-intent/runs/vlm-gensuite-coord}"
OUT_DIR="${HOME}/data/low-dim-intent/runs/vlm-gen-suite-job${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}/videos" "${OUT_DIR}/eval_dark200" "${OUT_DIR}/eval_casual" "${COORD_DIR}" logs
cp "${SLURM_SUBMIT_DIR}"/{record_vlm_gen_demo.py,eval_c2_isaac_closedloop.py,51_vlm_gen_suite_t4.sbatch} "${OUT_DIR}/"

ENDPOINT_FILE="${COORD_DIR}/endpoint.txt"
DONE_FLAG="${COORD_DIR}/done.flag"
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
export Z1_IMAGE="512" Z1_FOCAL="24" Z1_TARGET="0 -0.39 0.6"
mkdir -p "${XDG_CACHE_HOME}" "${HINGE_ASSET_DIR}"
unset UNIFIED_CYCLE UNIFIED_START_STATES FACE_TURN_START_STATES HINGE_STIFFNESS HINGE_DAMPING

echo "record_type=vlm_gen_suite job_id=${SLURM_JOB_ID} node=$(hostname -f) out_dir=${OUT_DIR}"

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
    echo "ERROR: Timed out waiting for vLLM server"
    exit 1
fi

cd "${ISAACLAB_DIR}"

echo "=== Part 1: Recording 3 hardcore generalization demo videos ==="
Z1_CAMERAS="c1:-50:38:0.33,s040:40:30:0.33,s220:-140:30:0.33,demo:-50:42:0.55" \
"${ISAAC_VENV}/bin/python" "${SLURM_SUBMIT_DIR}/record_vlm_gen_demo.py" \
    --headless \
    --checkpoint "${V4_CKPT}" \
    --vllm_url "${ENDPOINT}/v1/chat/completions" \
    --vllm_model "vlm_planner" \
    --only "gen3_dark_lowlight,gen4_ood_colour_rev,gen5_casual_spoken" \
    --output_dir "${OUT_DIR}/videos" \
    2>&1 | tee "${OUT_DIR}/part1_videos.log"

echo "=== Part 2a: 32-ep Quantitative Eval — True Dark Low-Light (intensity=200) ==="
Z1_CAMERAS="c1:-50:38:0.33,s040:40:30:0.33,s220:-140:30:0.33" \
"${ISAAC_VENV}/bin/python" "${SLURM_SUBMIT_DIR}/eval_c2_isaac_closedloop.py" \
    --headless \
    --checkpoint "${V4_CKPT}" \
    --vllm_url "${ENDPOINT}/v1/chat/completions" \
    --vllm_model "vlm_planner" \
    --num_envs 8 --batches_per_level 1 --levels "1,2,3,4" \
    --light_intensity 200 --light_color 0.75 0.75 0.75 \
    --prompt_style standard \
    --output_dir "${OUT_DIR}/eval_dark200" \
    2>&1 | tee "${OUT_DIR}/part2a_dark200.log"

echo "=== Part 2b: 32-ep Quantitative Eval — Pure Casual Spoken Prompts (No Schema Hint) ==="
Z1_CAMERAS="c1:-50:38:0.33,s040:40:30:0.33,s220:-140:30:0.33" \
"${ISAAC_VENV}/bin/python" "${SLURM_SUBMIT_DIR}/eval_c2_isaac_closedloop.py" \
    --headless \
    --checkpoint "${V4_CKPT}" \
    --vllm_url "${ENDPOINT}/v1/chat/completions" \
    --vllm_model "vlm_planner" \
    --num_envs 8 --batches_per_level 1 --levels "1,2,3,4" \
    --prompt_style casual \
    --output_dir "${OUT_DIR}/eval_casual" \
    2>&1 | tee "${OUT_DIR}/part2b_casual.log"

echo "vlm_gen_suite_status=PASS"
