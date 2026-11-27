#!/usr/bin/env bash
#SBATCH --job-name=p2-text-plan
#SBATCH --partition=gpu,gpu-long
#SBATCH --gpus=h100-47:1
#SBATCH --exclude=xgpi15,xgpj0
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Priority 2.2 (Z2 / Plan A zero-shot baseline): Privileged text-state multi-turn planning
# evaluation of Qwen3-VL-8B-Instruct across L = 1, 2, 3, 4 (50 tasks/level x greedy + G=8 temp=1.0 GRPO group).

set -Eeuo pipefail

VENV_DIR="${HOME}/software/vlm-stack/env_vllm"
HF_DIR="${HOME}/data/low-dim-intent/hf"
MODEL="${P2_MODEL:-Qwen3-VL-8B-Instruct}"
TASKS_PER_LEVEL="${P2_TASKS_PER_LEVEL:-50}"
GROUP_SIZE="${P2_GROUP_SIZE:-8}"
GPU_MEM_GIB="${P2_GPU_MEM_GIB:-40}"
PLAN_VARIANT="${P2_PLAN_VARIANT:-numeric}"
LEVELS="${P2_LEVELS:-1,2,3,4}"
TAG="${P2_TAG:-z2-${PLAN_VARIANT}}"
OUT_DIR="${HOME}/data/low-dim-intent/runs/p2-text-plan-${TAG}-${MODEL}-job${SLURM_JOB_ID}"

mkdir -p "${OUT_DIR}" logs
cp "${SLURM_SUBMIT_DIR}"/{cube_state_graph.py,p2_text_planner_eval.py,40_p2_text_planner.sbatch} "${OUT_DIR}/"

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_DIR}/cache"
export VLLM_CACHE_ROOT="${HOME}/data/low-dim-intent/cache/vllm"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

echo "record_type=p2_text_planner_job job_id=${SLURM_JOB_ID} node=$(hostname -f) timestamp=$(date --iso-8601=seconds)"
echo "gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | paste -sd ';')"
echo "model=${MODEL} variant=${PLAN_VARIANT} levels=${LEVELS} tasks_per_level=${TASKS_PER_LEVEL} group_size=${GROUP_SIZE} out_dir=${OUT_DIR}"

"${VENV_DIR}/bin/python" "${SLURM_SUBMIT_DIR}/p2_text_planner_eval.py" \
    --model "${HF_DIR}/${MODEL}" \
    --output_dir "${OUT_DIR}" \
    --plan_variant "${PLAN_VARIANT}" \
    --levels "${LEVELS}" \
    --tasks_per_level "${TASKS_PER_LEVEL}" \
    --group_size "${GROUP_SIZE}" \
    --gpu_mem_gib "${GPU_MEM_GIB}" \
    2>&1 | tee "${OUT_DIR}/eval.log"

grep -q "p2_text_planner_eval_status=PASS" "${OUT_DIR}/eval.log"
echo "p2_text_planner_job_status=PASS"
