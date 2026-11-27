#!/usr/bin/env bash
#SBATCH --job-name=bench-vlm
#SBATCH --partition=gpu
#SBATCH --gpus=h100-47:1
#SBATCH --time=02:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Public-benchmark pipeline check (bench_vlm_eval.py) for Qwen3-VL-8B-Instruct through the Z1 vLLM path.
# Optional env: BENCH_MODEL (default Qwen3-VL-8B-Instruct), BENCH_LIMIT (smoke test), BENCH_LIST, BENCH_DECODINGS,
# BENCH_TAG, BENCH_GPU_MEM_GIB (default 40 for a shared h100-47 slot).

set -Eeuo pipefail
VENV_DIR="${HOME}/software/vlm-stack/env_vllm"
HF_DIR="${HOME}/data/low-dim-intent/hf"
model="${BENCH_MODEL:-Qwen3-VL-8B-Instruct}"
model_path="${BENCH_MODEL_PATH:-${HF_DIR}/${model}}"
output_dir="${HOME}/data/low-dim-intent/runs/bench-${BENCH_TAG:-v1}-${model}-job${SLURM_JOB_ID}"
mkdir -p "${output_dir}"
cp "${SLURM_SUBMIT_DIR}"/{bench_vlm_eval.py,mcq_common.py,37_bench_vlm_eval.sbatch} "${output_dir}/"
export PYTHONUNBUFFERED=1 HF_HOME="${HF_DIR}/cache" VLLM_CACHE_ROOT="${HOME}/data/low-dim-intent/cache/vllm"
echo "record_type=bench_vlm_eval_job job_id=${SLURM_JOB_ID} node=$(hostname -f) timestamp=$(date --iso-8601=seconds)"
echo "gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | paste -sd ';')"
echo "model=${model} model_path=${model_path} output_dir=${output_dir}"

extra=(--gpu_mem_gib "${BENCH_GPU_MEM_GIB:-40}")
[[ -n "${BENCH_LIMIT:-}" ]] && extra+=(--limit "${BENCH_LIMIT}")
[[ -n "${BENCH_LIST:-}" ]] && extra+=(--benchmarks "${BENCH_LIST}")
[[ -n "${BENCH_DECODINGS:-}" ]] && extra+=(--decodings "${BENCH_DECODINGS}")
"${VENV_DIR}/bin/python" "${SLURM_SUBMIT_DIR}/bench_vlm_eval.py" --model "${model_path}" \
    --data_dir "${HF_DIR}/datasets" --output_dir "${output_dir}" "${extra[@]}" 2>&1 | tee "${output_dir}/eval.log"
grep -q "bench_vlm_eval_status=PASS" "${output_dir}/eval.log"
