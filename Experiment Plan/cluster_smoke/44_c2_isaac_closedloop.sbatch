#!/usr/bin/env bash
#SBATCH --job-name=c1-closedloop-both
#SBATCH --partition=gpu
#SBATCH --constraint=h200
#SBATCH --gpus=h200-141:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Line 1: Runs BOTH closed-loop evaluations on a single H200-141 GPU:
#   1a. 200 frozen test tasks multi-turn closed-loop on real montage image pool (vLLM)
#   1b. Real 60 Hz Isaac Lab Unified v4 + live 3-camera rendering + localhost vLLM HTTP closed-loop (128 physical eps)

set -Eeuo pipefail

VLLM_VENV="${HOME}/software/vlm-stack/env_vllm"
ISAAC_VENV="${HOME}/software/isaaclab-stack/env_isaaclab"
ISAACLAB_DIR="${HOME}/software/isaaclab-stack/IsaacLab"
HF_DIR="${HOME}/data/low-dim-intent/hf"

MODEL_DIR="${CL_MODEL_DIR:-${HOME}/data/low-dim-intent/runs/c1-sft-vision-full-job1173965/merged_hf}"
TAG="${CL_TAG:-c1-vision-full}"
V4_CKPT="${HOME}/data/low-dim-intent/checkpoints/expert-unified-cap0.05_hold10_v4-seed0/frozen/expert-unified-v4-seed0.pth"
FROZEN_TASKS="${HOME}/data/low-dim-intent/runs/p2-text-plan-z2-relational-Qwen3-VL-8B-Instruct-job1169616/tasks.json"
FRAMES="${HOME}/data/low-dim-intent/runs/z1-questions/z1_frames_v2.jsonl"
TRAIN_POOL="${HOME}/data/low-dim-intent/datasets/c1-sft-montage-job1173524/state_image_pool.json"
PORT=$(( 18000 + (SLURM_JOB_ID % 1000) ))

OUT_DIR="${HOME}/data/low-dim-intent/runs/closedloop-${TAG}-job${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}" logs
cp "${SLURM_SUBMIT_DIR}"/{eval_c2_closedloop_pool.py,eval_c2_isaac_closedloop.py,44_c2_isaac_closedloop.sbatch} "${OUT_DIR}/"

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_DIR}/cache"
export VLLM_CACHE_ROOT="${HOME}/data/low-dim-intent/cache/vllm"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

echo "record_type=closedloop_eval job_id=${SLURM_JOB_ID} node=$(hostname -f) timestamp=$(date --iso-8601=seconds)"
echo "model=${MODEL_DIR} tag=${TAG} port=${PORT} out_dir=${OUT_DIR}"

echo "=== Part 1a: 200 Frozen Test Tasks Closed-Loop (Image Pool) ==="
"${VLLM_VENV}/bin/python" "${SLURM_SUBMIT_DIR}/eval_c2_closedloop_pool.py" \
    --model "${MODEL_DIR}" \
    --tasks "${FROZEN_TASKS}" \
    --test_frames "${FRAMES}" \
    --train_pool "${TRAIN_POOL}" \
    --output_dir "${OUT_DIR}/pool_200" \
    --max_tokens 512 \
    --gpu_mem_gib 45 \
    2>&1 | tee "${OUT_DIR}/pool_200.log"
grep -q "eval_c2_closedloop_pool_status=PASS" "${OUT_DIR}/pool_200.log"

echo "=== Part 1b: Real 60 Hz Isaac Lab Unified v4 Online Rendering Closed-Loop ==="
VLLM_PID=""
cleanup() {
    if [[ -n "${VLLM_PID}" ]]; then
        kill "${VLLM_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

"${VLLM_VENV}/bin/vllm" serve "${MODEL_DIR}" \
    --served-model-name "vlm_planner" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --gpu-memory-utilization 0.25 \
    --max-model-len 8192 \
    --limit-mm-per-prompt '{"image": 1}' \
    --disable-log-requests \
    > "${OUT_DIR}/vllm_server.log" 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM server on port ${PORT} (pid=${VLLM_PID})..."
for i in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "vLLM server ready after $((i * 2))s"
        break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "ERROR: vLLM server exited prematurely!"
        tail -n 40 "${OUT_DIR}/vllm_server.log"
        exit 1
    fi
    sleep 2
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null

# Source unified v4 + Z1 rendering environment variables (matching 33_z1_capture.sbatch)
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

(
    cd "${ISAACLAB_DIR}"
    "${ISAAC_VENV}/bin/python" "${SLURM_SUBMIT_DIR}/eval_c2_isaac_closedloop.py" \
        --headless \
        --checkpoint "${V4_CKPT}" \
        --vllm_url "http://127.0.0.1:${PORT}/v1/chat/completions" \
        --vllm_model "vlm_planner" \
        --num_envs 16 \
        --batches_per_level 2 \
        --levels "1,2,3,4" \
        --max_round_steps 120 \
        --output_dir "${OUT_DIR}/isaac_128" \
        2>&1 | tee "${OUT_DIR}/isaac_128.log"
)
grep -q "eval_c2_isaac_closedloop_status=PASS" "${OUT_DIR}/isaac_128.log"

echo "closedloop_both_status=PASS"
