#!/usr/bin/env bash
#SBATCH --job-name=c2-visual-grpo
#SBATCH --partition=gpu,gpu-long
#SBATCH --gpus=h100-47:1
#SBATCH --exclude=xgpi15,xgpj0
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -Eeuo pipefail

VENV_DIR="${HOME}/software/vlm-stack/env_vllm"
HF_DIR="${HOME}/data/low-dim-intent/hf"
SFT_MODEL="${C2_SFT_MODEL:-${HOME}/data/low-dim-intent/runs/c1-sft-vision-full-job1173965/merged_hf}"
TRAIN_POOL="${HOME}/data/low-dim-intent/datasets/c1-sft-montage-job1173524/state_image_pool.json"
FROZEN_TASKS="${HOME}/data/low-dim-intent/runs/p2-text-plan-z2-relational-Qwen3-VL-8B-Instruct-job1169616/tasks.json"
FRAMES="${HOME}/data/low-dim-intent/runs/z1-questions/z1_frames_v2.jsonl"
REWARD_MODE="${C2_REWARD_MODE:?Set C2_REWARD_MODE to grounded or terminal}"
STEPS="${C2_STEPS:-60}"
VLLM_GIB="${C2_VLLM_GIB:-40}"

RUN_DIR="${HOME}/data/low-dim-intent/runs/c2-grpo-${REWARD_MODE}-job${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}" logs
cp "${SLURM_SUBMIT_DIR}"/{train_c2_visual_grpo.py,eval_c2_closedloop_pool.py,eval_c1_openloop_plan.py,43_c2_visual_grpo.sbatch} "${RUN_DIR}/"

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_DIR}/cache"
export VLLM_CACHE_ROOT="${HOME}/data/low-dim-intent/cache/vllm"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

echo "record_type=c2_visual_grpo job_id=${SLURM_JOB_ID} node=$(hostname -f) timestamp=$(date --iso-8601=seconds)"
echo "sft_model=${SFT_MODEL} reward_mode=${REWARD_MODE} steps=${STEPS} run_dir=${RUN_DIR}"

echo "=== Step 1: Training Stage C2 Visual GRPO (${REWARD_MODE}) ==="
"${VENV_DIR}/bin/python" "${SLURM_SUBMIT_DIR}/train_c2_visual_grpo.py" \
    --model "${SFT_MODEL}" \
    --train_pool "${TRAIN_POOL}" \
    --exclude_tasks "${FROZEN_TASKS}" \
    --output_dir "${RUN_DIR}" \
    --reward_mode "${REWARD_MODE}" \
    --steps "${STEPS}" \
    --tasks_per_step 4 \
    --group_size 6 \
    --lr 2e-5 \
    --kl_beta 0.02 \
    2>&1 | tee "${RUN_DIR}/train_grpo.log"
grep -q "train_c2_visual_grpo_status=PASS" "${RUN_DIR}/train_grpo.log"

echo "=== Step 2: Evaluating C2 on 200 Frozen Test Tasks (Closed-loop Image Pool) ==="
"${VENV_DIR}/bin/python" "${SLURM_SUBMIT_DIR}/eval_c2_closedloop_pool.py" \
    --model "${RUN_DIR}/merged_hf" \
    --tasks "${FROZEN_TASKS}" \
    --test_frames "${FRAMES}" \
    --train_pool "${TRAIN_POOL}" \
    --output_dir "${RUN_DIR}/eval_closedloop_pool" \
    --max_tokens 512 \
    --gpu_mem_gib "${VLLM_GIB}" \
    2>&1 | tee "${RUN_DIR}/eval_closedloop_pool.log"
grep -q "eval_c2_closedloop_pool_status=PASS" "${RUN_DIR}/eval_closedloop_pool.log"

echo "=== Step 3: Evaluating C2 on Open-loop 96-state + 360 BFS Planning ==="
"${VENV_DIR}/bin/python" "${SLURM_SUBMIT_DIR}/eval_c1_openloop_plan.py" \
    --model "${RUN_DIR}/merged_hf" \
    --frames "${FRAMES}" \
    --output_dir "${RUN_DIR}/eval_openloop_plan" \
    --mode vision_full \
    --plans_per_frame 2 \
    --max_tokens 512 \
    --gpu_mem_gib "${VLLM_GIB}" \
    2>&1 | tee "${RUN_DIR}/eval_openloop_plan.log"
grep -q "eval_c1_openloop_plan_status=PASS" "${RUN_DIR}/eval_openloop_plan.log"

echo "c2_visual_grpo_pipeline_status=PASS"
