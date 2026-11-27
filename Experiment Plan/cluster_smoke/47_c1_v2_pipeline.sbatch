#!/usr/bin/env bash
#SBATCH --job-name=c1-v2-pipeline
#SBATCH --partition=gpu,gpu-long
#SBATCH --gpus=h100-47:1
#SBATCH --exclude=xgpi15,xgpj0
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# C1-v2 unified SFT pipeline (multi-template bidirectional + occlusion CBD + 30% color-aug + cyclic turn CoT, 1 epoch):
#   1. Build C1-v2 dataset (reusing canonical montage PNGs from job 1173524, strictly excluding test episodes & 200 pairs)
#   2. Train Qwen3-VL-8B-Instruct LoRA SFT (1 epoch, lr=1.0e-4, dropout=0.05) and merge to merged_hf/
#   3. Run full 6-part verification suite:
#      (a) Canonical Z1 MCQ + unanswerable CBD check
#      (b) OOD reverse_cycle Z1 MCQ
#      (c) OOD novel_palette (orange/cyan/purple/pink) Z1 MCQ
#      (d) Unseen prompt template question_set=v3 Z1 MCQ
#      (e) Open-loop 96-state + 360 BFS planning (verify turn(+90/-90) fix!)
#      (f) Closed-loop 200 frozen test tasks (verify L2 rounds ratio improvement!)

set -Eeuo pipefail

VENV_DIR="${HOME}/software/vlm-stack/env_vllm"
HF_DIR="${HOME}/data/low-dim-intent/hf"
BASE_MODEL="${HF_DIR}/Qwen3-VL-8B-Instruct"
TRAIN_RUNS="${HOME}/data/low-dim-intent/runs/z1-c1-3cam-job1173524_*"
FRAMES="${HOME}/data/low-dim-intent/runs/z1-questions/z1_frames_v2.jsonl"
FROZEN_TASKS="${HOME}/data/low-dim-intent/runs/p2-text-plan-z2-relational-Qwen3-VL-8B-Instruct-job1169616/tasks.json"
TRAIN_POOL="${HOME}/data/low-dim-intent/datasets/c1-sft-montage-job1173524/state_image_pool.json"
REUSE_IMGS="${HOME}/data/low-dim-intent/datasets/c1-sft-montage-job1173524/images"

DATASET_DIR="${HOME}/data/low-dim-intent/datasets/c1-v2-montage-job${SLURM_JOB_ID}"
RUN_DIR="${HOME}/data/low-dim-intent/runs/c1-v2-sft-job${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}" "${DATASET_DIR}" logs
cp "${SLURM_SUBMIT_DIR}"/{build_c1_v2_dataset.py,train_qwen3vl_lora_sft.py,z1_mcq_eval.py,eval_c1_openloop_plan.py,eval_c2_closedloop_pool.py,47_c1_v2_pipeline.sbatch} "${RUN_DIR}/"

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_DIR}/cache"
export VLLM_CACHE_ROOT="${HOME}/data/low-dim-intent/cache/vllm"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

echo "record_type=c1_v2_pipeline job_id=${SLURM_JOB_ID} node=$(hostname -f) timestamp=$(date --iso-8601=seconds)"
echo "dataset_dir=${DATASET_DIR} run_dir=${RUN_DIR}"

echo "=== Step 1: Building C1-v2 SFT dataset ==="
"${VENV_DIR}/bin/python" "${SLURM_SUBMIT_DIR}/build_c1_v2_dataset.py" \
    --runs "${TRAIN_RUNS}" \
    --exclude_frames "${FRAMES}" \
    --exclude_tasks "${FROZEN_TASKS}" \
    --reuse_img_dir "${REUSE_IMGS}" \
    --output_dir "${DATASET_DIR}" \
    2>&1 | tee "${RUN_DIR}/build_dataset.log"
grep -q "build_c1_v2_dataset_status=PASS" "${RUN_DIR}/build_dataset.log"

echo "=== Step 2: Training C1-v2 LoRA SFT (1 epoch, lr=1.0e-4) ==="
"${VENV_DIR}/bin/python" "${SLURM_SUBMIT_DIR}/train_qwen3vl_lora_sft.py" \
    --model "${BASE_MODEL}" \
    --train_jsonl "${DATASET_DIR}/sft_c1_v2.jsonl" \
    --output_dir "${RUN_DIR}" \
    --epochs 1 \
    --grad_accum 8 \
    --lr 1.0e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    2>&1 | tee "${RUN_DIR}/train.log"
grep -q "train_qwen3vl_lora_sft_status=PASS" "${RUN_DIR}/train.log"

VLLM_GIB=40
echo "=== Step 3a: Canonical Z1 MCQ (with hand + @nohand + CBD) ==="
"${VENV_DIR}/bin/python" "${SLURM_SUBMIT_DIR}/z1_mcq_eval.py" \
    --model "${RUN_DIR}/merged_hf" --frames "${FRAMES}" \
    --output_dir "${RUN_DIR}/eval_canonical" --splits test \
    --conditions "c1+s040+s220,c1+s040+s220@nohand" \
    --visual_mode montage --ood_color_mode none --max_tokens 512 --gpu_mem_gib "${VLLM_GIB}" \
    2>&1 | tee "${RUN_DIR}/eval_canonical.log"
grep -q "z1_mcq_eval_status=PASS" "${RUN_DIR}/eval_canonical.log"

echo "=== Step 3b: OOD reverse_cycle Z1 MCQ ==="
"${VENV_DIR}/bin/python" "${SLURM_SUBMIT_DIR}/z1_mcq_eval.py" \
    --model "${RUN_DIR}/merged_hf" --frames "${FRAMES}" \
    --output_dir "${RUN_DIR}/eval_ood_rev" --splits test \
    --conditions "c1+s040+s220,c1+s040+s220@nohand" \
    --visual_mode montage --ood_color_mode reverse_cycle --max_tokens 512 --gpu_mem_gib "${VLLM_GIB}" \
    2>&1 | tee "${RUN_DIR}/eval_ood_rev.log"
grep -q "z1_mcq_eval_status=PASS" "${RUN_DIR}/eval_ood_rev.log"

echo "=== Step 3c: OOD novel_palette (orange/cyan/purple/pink) Z1 MCQ ==="
"${VENV_DIR}/bin/python" "${SLURM_SUBMIT_DIR}/z1_mcq_eval.py" \
    --model "${RUN_DIR}/merged_hf" --frames "${FRAMES}" \
    --output_dir "${RUN_DIR}/eval_ood_novel" --splits test \
    --conditions "c1+s040+s220,c1+s040+s220@nohand" \
    --visual_mode montage --ood_color_mode novel_palette --max_tokens 512 --gpu_mem_gib "${VLLM_GIB}" \
    2>&1 | tee "${RUN_DIR}/eval_ood_novel.log"
grep -q "z1_mcq_eval_status=PASS" "${RUN_DIR}/eval_ood_novel.log"

echo "=== Step 3d: Unseen Prompt Template (question_set=v3) Z1 MCQ ==="
"${VENV_DIR}/bin/python" "${SLURM_SUBMIT_DIR}/z1_mcq_eval.py" \
    --model "${RUN_DIR}/merged_hf" --frames "${FRAMES}" \
    --output_dir "${RUN_DIR}/eval_v3_prompt" --splits test \
    --conditions "c1+s040+s220" \
    --visual_mode montage --question_set v3 --ood_color_mode none --max_tokens 512 --gpu_mem_gib "${VLLM_GIB}" \
    2>&1 | tee "${RUN_DIR}/eval_v3_prompt.log"
grep -q "z1_mcq_eval_status=PASS" "${RUN_DIR}/eval_v3_prompt.log"

echo "=== Step 3e: Open-loop 96-state + 360 BFS Planning ==="
"${VENV_DIR}/bin/python" "${SLURM_SUBMIT_DIR}/eval_c1_openloop_plan.py" \
    --model "${RUN_DIR}/merged_hf" --frames "${FRAMES}" \
    --output_dir "${RUN_DIR}/eval_openloop_plan" --mode vision_full \
    --plans_per_frame 2 --max_tokens 512 --gpu_mem_gib "${VLLM_GIB}" \
    2>&1 | tee "${RUN_DIR}/eval_openloop_plan.log"
grep -q "eval_c1_openloop_plan_status=PASS" "${RUN_DIR}/eval_openloop_plan.log"

echo "=== Step 3f: Closed-loop 200 Frozen Test Tasks ==="
"${VENV_DIR}/bin/python" "${SLURM_SUBMIT_DIR}/eval_c2_closedloop_pool.py" \
    --model "${RUN_DIR}/merged_hf" --tasks "${FROZEN_TASKS}" \
    --test_frames "${FRAMES}" --train_pool "${TRAIN_POOL}" \
    --output_dir "${RUN_DIR}/eval_closedloop_pool" --max_tokens 512 --gpu_mem_gib "${VLLM_GIB}" \
    2>&1 | tee "${RUN_DIR}/eval_closedloop_pool.log"
grep -q "eval_c2_closedloop_pool_status=PASS" "${RUN_DIR}/eval_closedloop_pool.log"

echo "c1_v2_pipeline_status=PASS"
