#!/usr/bin/env bash
#SBATCH --job-name=c1-sft-pipeline
#SBATCH --partition=gpu,gpu-long
#SBATCH --gpus=h100-47:1
#SBATCH --exclude=xgpi15,xgpj0
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Scheme C (Stage C1) & Dactyl Smart-Cube Fallback pipeline:
#   1. Build montage SFT datasets (sft_vision_full.jsonl + sft_smart_cube.jsonl + 96-state image pool)
#      strictly disjoint from the 180-frame z1_frames_v2 test split.
#   2. Train Qwen3-VL-8B-Instruct LoRA SFT (pure PyTorch + HF, no extra deps) and merge into HF dir.
#   3. Evaluate the merged model with vLLM on the held-out 180-frame z1_frames_v2 test split (montage mode,
#      ZERO SoM tags!) to verify fine-grained band/turned accuracy jump from 0.075 -> 0.85+.

set -Eeuo pipefail

VENV_DIR="${HOME}/software/vlm-stack/env_vllm"
HF_DIR="${HOME}/data/low-dim-intent/hf"
BASE_MODEL="${HF_DIR}/${C1_MODEL:-Qwen3-VL-8B-Instruct}"
TRAIN_RUNS="${C1_TRAIN_RUNS:-${HOME}/data/low-dim-intent/runs/z1-c1-train-job1173340_*}"
EXCLUDE_FRAMES="${HOME}/data/low-dim-intent/runs/z1-questions/z1_frames_v2.jsonl"
SFT_MODE="${C1_SFT_MODE:-vision_full}"   # vision_full (Scheme C) or smart_cube (Dactyl Fallback)
EPOCHS="${C1_EPOCHS:-2}"
MAX_FRAMES_PER_STATE="${C1_MAX_FRAMES_PER_STATE:-25}"
TAG="${C1_TAG:-${SFT_MODE}}"

DATASET_DIR="${C1_DATASET_DIR:-${HOME}/data/low-dim-intent/datasets/c1-sft-montage-job1173340}"
RUN_DIR="${HOME}/data/low-dim-intent/runs/c1-sft-${TAG}-job${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}" logs
cp "${SLURM_SUBMIT_DIR}"/{cube_state_graph.py,build_c1_sft_dataset.py,train_qwen3vl_lora_sft.py,z1_mcq_eval.py,mcq_common.py,41_c1_sft_pipeline.sbatch} "${RUN_DIR}/"

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_DIR}/cache"
export VLLM_CACHE_ROOT="${HOME}/data/low-dim-intent/cache/vllm"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

echo "record_type=c1_sft_pipeline_job job_id=${SLURM_JOB_ID} node=$(hostname -f) timestamp=$(date --iso-8601=seconds)"
echo "gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | paste -sd ';')"
echo "sft_mode=${SFT_MODE} dataset_dir=${DATASET_DIR} run_dir=${RUN_DIR}"

# Step 1: Build dataset once (skip if manifest.json already exists)
if [[ ! -f "${DATASET_DIR}/manifest.json" ]]; then
    echo "=== Step 1: Building montage SFT dataset ==="
    "${VENV_DIR}/bin/python" "${SLURM_SUBMIT_DIR}/build_c1_sft_dataset.py" \
        --runs "${TRAIN_RUNS}" \
        --exclude_frames "${EXCLUDE_FRAMES}" \
        --output_dir "${DATASET_DIR}" \
        --max_frames_per_state "${MAX_FRAMES_PER_STATE}" \
        --plans_per_frame 2 \
        2>&1 | tee "${RUN_DIR}/build_dataset.log"
    grep -q "build_c1_sft_dataset_status=PASS" "${RUN_DIR}/build_dataset.log"
else
    echo "=== Step 1: Reusing existing dataset at ${DATASET_DIR} ==="
fi

TRAIN_JSONL="${DATASET_DIR}/sft_${SFT_MODE}.jsonl"
test -f "${TRAIN_JSONL}"

# Step 2: LoRA SFT + merge into HF checkpoint
echo "=== Step 2: Training Qwen3-VL-8B-Instruct LoRA SFT (${SFT_MODE}) ==="
"${VENV_DIR}/bin/python" "${SLURM_SUBMIT_DIR}/train_qwen3vl_lora_sft.py" \
    --model "${BASE_MODEL}" \
    --train_jsonl "${TRAIN_JSONL}" \
    --output_dir "${RUN_DIR}" \
    --epochs "${EPOCHS}" \
    --grad_accum 8 \
    --lr 1.5e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    2>&1 | tee "${RUN_DIR}/train.log"
grep -q "train_qwen3vl_lora_sft_status=PASS" "${RUN_DIR}/train.log"

# Step 3: Evaluate merged model on held-out 180-frame z1_frames_v2 test split (canonical montage + OOD recolored montage!)
VLLM_GIB="${C1_VLLM_GIB:-40}"
echo "=== Step 3a: Evaluating merged model on held-out 180-frame Z1 test set (canonical montage) ==="
EVAL_DIR="${RUN_DIR}/eval_z1_v2_montage"
"${VENV_DIR}/bin/python" "${SLURM_SUBMIT_DIR}/z1_mcq_eval.py" \
    --model "${RUN_DIR}/merged_hf" \
    --frames "${EXCLUDE_FRAMES}" \
    --output_dir "${EVAL_DIR}" \
    --splits test \
    --conditions "c1+s040+s220,c1+s040+s220@nohand" \
    --visual_mode montage \
    --ood_color_mode none \
    --max_tokens 512 \
    --gpu_mem_gib "${VLLM_GIB}" \
    2>&1 | tee "${RUN_DIR}/eval_z1.log"
grep -q "z1_mcq_eval_status=PASS" "${RUN_DIR}/eval_z1.log"

echo "=== Step 3b: Evaluating merged model on OOD Recolored Cube (reverse_cycle: green<->blue swapped) ==="
EVAL_OOD_DIR="${RUN_DIR}/eval_z1_v2_montage_ood"
"${VENV_DIR}/bin/python" "${SLURM_SUBMIT_DIR}/z1_mcq_eval.py" \
    --model "${RUN_DIR}/merged_hf" \
    --frames "${EXCLUDE_FRAMES}" \
    --output_dir "${EVAL_OOD_DIR}" \
    --splits test \
    --conditions "c1+s040+s220,c1+s040+s220@nohand" \
    --visual_mode montage \
    --ood_color_mode reverse_cycle \
    --max_tokens 512 \
    --gpu_mem_gib "${VLLM_GIB}" \
    2>&1 | tee "${RUN_DIR}/eval_z1_ood.log"
grep -q "z1_mcq_eval_status=PASS" "${RUN_DIR}/eval_z1_ood.log"

echo "c1_sft_pipeline_status=PASS"
