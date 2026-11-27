#!/usr/bin/env bash
#SBATCH --job-name=c1-boundary-plan-eval
#SBATCH --partition=gpu,gpu-long
#SBATCH --gpus=h100-47:1
#SBATCH --exclude=xgpi15,xgpj0
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -Eeuo pipefail

VENV_DIR="${HOME}/software/vlm-stack/env_vllm"
HF_DIR="${HOME}/data/low-dim-intent/hf"
FRAMES="${HOME}/data/low-dim-intent/runs/z1-questions/z1_frames_v2.jsonl"
MODEL_DIR="${C1_EVAL_MODEL:?Set C1_EVAL_MODEL to merged_hf path}"
EVAL_MODE="${C1_EVAL_MODE:?Set C1_EVAL_MODE to vision_full or color_invariant}"
RUN_EXTRA_BOUNDARIES="${C1_EXTRA_BOUNDARIES:-1}"
VLLM_GIB="${C1_VLLM_GIB:-40}"

OUT_DIR="${HOME}/data/low-dim-intent/runs/c1-boundary-plan-${EVAL_MODE}-job${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}" logs
cp "${SLURM_SUBMIT_DIR}"/{z1_mcq_eval.py,eval_c1_openloop_plan.py,42_c1_boundary_and_plan_eval.sbatch} "${OUT_DIR}/"

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_DIR}/cache"
export VLLM_CACHE_ROOT="${HOME}/data/low-dim-intent/cache/vllm"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

echo "record_type=c1_boundary_plan_eval job_id=${SLURM_JOB_ID} node=$(hostname -f) timestamp=$(date --iso-8601=seconds)"
echo "model=${MODEL_DIR} mode=${EVAL_MODE} extra_boundaries=${RUN_EXTRA_BOUNDARIES} out_dir=${OUT_DIR}"

echo "=== 1. Novel Unseen Color Palette (orange/cyan/purple/pink) ==="
"${VENV_DIR}/bin/python" "${SLURM_SUBMIT_DIR}/z1_mcq_eval.py" \
    --model "${MODEL_DIR}" \
    --frames "${FRAMES}" \
    --output_dir "${OUT_DIR}/mcq_novel_palette" \
    --splits test \
    --conditions "c1+s040+s220,c1+s040+s220@nohand" \
    --visual_mode montage \
    --ood_color_mode novel_palette \
    --max_tokens 512 \
    --gpu_mem_gib "${VLLM_GIB}" \
    2>&1 | tee "${OUT_DIR}/mcq_novel_palette.log"
grep -q "z1_mcq_eval_status=PASS" "${OUT_DIR}/mcq_novel_palette.log"

if [[ "${RUN_EXTRA_BOUNDARIES}" == "1" ]]; then
    echo "=== 2. Unseen Camera Viewpoints (c1, top, s040+s220, s130+top) ==="
    "${VENV_DIR}/bin/python" "${SLURM_SUBMIT_DIR}/z1_mcq_eval.py" \
        --model "${MODEL_DIR}" \
        --frames "${FRAMES}" \
        --output_dir "${OUT_DIR}/mcq_unseen_views" \
        --splits test \
        --conditions "c1,top,s040+s220,s130+top" \
        --visual_mode montage \
        --ood_color_mode none \
        --max_tokens 512 \
        --gpu_mem_gib "${VLLM_GIB}" \
        2>&1 | tee "${OUT_DIR}/mcq_unseen_views.log"
    grep -q "z1_mcq_eval_status=PASS" "${OUT_DIR}/mcq_unseen_views.log"

    echo "=== 3. Unseen Prompt Template (question_set=v3) ==="
    "${VENV_DIR}/bin/python" "${SLURM_SUBMIT_DIR}/z1_mcq_eval.py" \
        --model "${MODEL_DIR}" \
        --frames "${FRAMES}" \
        --output_dir "${OUT_DIR}/mcq_v3_prompt" \
        --splits test \
        --conditions "c1+s040+s220" \
        --visual_mode montage \
        --question_set v3 \
        --ood_color_mode none \
        --max_tokens 512 \
        --gpu_mem_gib "${VLLM_GIB}" \
        2>&1 | tee "${OUT_DIR}/mcq_v3_prompt.log"
    grep -q "z1_mcq_eval_status=PASS" "${OUT_DIR}/mcq_v3_prompt.log"
fi

echo "=== 4. Open-loop 96-state Perception + BFS Planning ==="
"${VENV_DIR}/bin/python" "${SLURM_SUBMIT_DIR}/eval_c1_openloop_plan.py" \
    --model "${MODEL_DIR}" \
    --frames "${FRAMES}" \
    --output_dir "${OUT_DIR}/openloop_plan" \
    --mode "${EVAL_MODE}" \
    --plans_per_frame 2 \
    --max_tokens 512 \
    --gpu_mem_gib "${VLLM_GIB}" \
    2>&1 | tee "${OUT_DIR}/openloop_plan.log"
grep -q "eval_c1_openloop_plan_status=PASS" "${OUT_DIR}/openloop_plan.log"

echo "c1_boundary_plan_eval_status=PASS"
