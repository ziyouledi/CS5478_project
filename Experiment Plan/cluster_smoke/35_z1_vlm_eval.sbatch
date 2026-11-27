#!/usr/bin/env bash
#SBATCH --job-name=z1-vlm
#SBATCH --partition=gpu,gpu-long
#SBATCH --gpus=h100-47:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Z1 zero-shot VLM evaluation (z1_vlm_eval.py), one model per array task:
#   0 Qwen3-VL-8B-Thinking   1 Qwen3-VL-8B-Instruct   2 Qwen2.5-VL-7B-Instruct
# Env from 34_install_vlm_stack.sbatch. GPUs here are requested by type (a bare --gpus=1 only matches T4 / Titan nodes). Optional: Z1_QUESTIONS, Z1_LIMIT (smoke test), Z1_CONDITIONS,
# Z1_SAMPLES, Z1_TAG, Z1_PROMPT (full | colour).

set -Eeuo pipefail
VENV_DIR="${HOME}/software/vlm-stack/env_vllm"
HF_DIR="${HOME}/data/low-dim-intent/hf"
models=(Qwen3-VL-8B-Thinking Qwen3-VL-8B-Instruct Qwen2.5-VL-7B-Instruct)
# Instruct models reason at length unprompted: 40-50% of completions hit 2048 tokens in smoke test 1164403.
max_tokens=(16384 4096 4096)
task_id="${SLURM_ARRAY_TASK_ID:-0}"
model="${models[${task_id}]}"
questions="${Z1_QUESTIONS:-${HOME}/data/low-dim-intent/runs/z1-questions/z1_questions_v1.jsonl}"
output_dir="${HOME}/data/low-dim-intent/runs/z1-vlm-${Z1_TAG:-v1}-${model}-job${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${task_id}"
mkdir -p "${output_dir}"
cp "${SLURM_SUBMIT_DIR}"/{z1_vlm_eval.py,35_z1_vlm_eval.sbatch} "${output_dir}/"

export PYTHONUNBUFFERED=1 HF_HOME="${HF_DIR}/cache" VLLM_CACHE_ROOT="${HOME}/data/low-dim-intent/cache/vllm"
echo "record_type=z1_vlm_eval_job"
echo "timestamp=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID} array_task=${task_id}"
echo "node=$(hostname -f)"
echo "gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | paste -sd ';')"
echo "model=${model}"
echo "questions=${questions} sha256=$(sha256sum "${questions}" | cut -d' ' -f1)"
echo "output_dir=${output_dir}"

extra=()
[[ -n "${Z1_LIMIT:-}" ]] && extra+=(--limit "${Z1_LIMIT}")
[[ -n "${Z1_CONDITIONS:-}" ]] && extra+=(--conditions "${Z1_CONDITIONS}")
[[ -n "${Z1_SAMPLES:-}" ]] && extra+=(--samples "${Z1_SAMPLES}")
[[ -n "${Z1_PROMPT:-}" ]] && extra+=(--prompt "${Z1_PROMPT}")
# Shared h100-47 slot: fixed 40 GiB budget. A full card (Z1_GPU_MEM_GIB=0, e.g. h100-96 for the Thinking model): 90%.
extra+=(--gpu_mem_gib "${Z1_GPU_MEM_GIB:-40}")
"${VENV_DIR}/bin/python" "${SLURM_SUBMIT_DIR}/z1_vlm_eval.py" \
    --model "${HF_DIR}/${model}" --questions "${questions}" --output_dir "${output_dir}" \
    --max_tokens "${max_tokens[${task_id}]}" "${extra[@]}" 2>&1 | tee "${output_dir}/eval.log"
grep -q "z1_vlm_eval_status=PASS" "${output_dir}/eval.log"
