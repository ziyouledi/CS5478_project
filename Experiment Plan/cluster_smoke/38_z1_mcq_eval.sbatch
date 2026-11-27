#!/usr/bin/env bash
#SBATCH --job-name=z1-mcq
#SBATCH --partition=gpu-long
#SBATCH --gpus=h100-47:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Z1 v2 standard multiple-choice evaluation (z1_mcq_eval.py) of Qwen3-VL-8B-Instruct on the frames from
# z1_build_mcq.py. Optional env: Z1_MODEL, Z1_FRAMES, Z1_LIMIT (smoke test), Z1_CONDITIONS, Z1_SPLITS, Z1_TAG,
# Z1_GPU_MEM_GIB (default 40 for a shared h100-47 slot).

set -Eeuo pipefail
VENV_DIR="${HOME}/software/vlm-stack/env_vllm"
HF_DIR="${HOME}/data/low-dim-intent/hf"
model="${Z1_MODEL:-Qwen3-VL-8B-Instruct}"
frames="${Z1_FRAMES:-${HOME}/data/low-dim-intent/runs/z1-questions/z1_frames_v2.jsonl}"
output_dir="${HOME}/data/low-dim-intent/runs/z1-mcq-${Z1_TAG:-v2}-${model}-job${SLURM_JOB_ID}"
mkdir -p "${output_dir}"
cp "${SLURM_SUBMIT_DIR}"/{z1_mcq_eval.py,mcq_common.py,z1_build_mcq.py,38_z1_mcq_eval.sbatch} "${output_dir}/"
export PYTHONUNBUFFERED=1 HF_HOME="${HF_DIR}/cache" VLLM_CACHE_ROOT="${HOME}/data/low-dim-intent/cache/vllm"
echo "record_type=z1_mcq_eval_job job_id=${SLURM_JOB_ID} node=$(hostname -f) timestamp=$(date --iso-8601=seconds)"
echo "gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | paste -sd ';')"
echo "model=${model} frames=${frames} sha256=$(sha256sum "${frames}" | cut -d' ' -f1) output_dir=${output_dir}"

extra=(--gpu_mem_gib "${Z1_GPU_MEM_GIB:-40}")
[[ -n "${Z1_LIMIT:-}" ]] && extra+=(--limit "${Z1_LIMIT}")
[[ -n "${Z1_CONDITIONS:-}" ]] && extra+=(--conditions "${Z1_CONDITIONS}")
[[ -n "${Z1_SPLITS:-}" ]] && extra+=(--splits "${Z1_SPLITS}")
[[ -n "${Z1_VISUAL_MODE:-}" ]] && extra+=(--visual_mode "${Z1_VISUAL_MODE}")
[[ -n "${Z1_CROP_SIZE:-}" ]] && extra+=(--crop_size "${Z1_CROP_SIZE}")
[[ -n "${Z1_CROP_PAD:-}" ]] && extra+=(--crop_pad "${Z1_CROP_PAD}")
[[ -n "${Z1_QUESTION_SET:-}" ]] && extra+=(--question_set "${Z1_QUESTION_SET}")
"${VENV_DIR}/bin/python" "${SLURM_SUBMIT_DIR}/z1_mcq_eval.py" --model "${HF_DIR}/${model}" --frames "${frames}" \
    --output_dir "${output_dir}" "${extra[@]}" 2>&1 | tee "${output_dir}/eval.log"
grep -q "z1_mcq_eval_status=PASS" "${output_dir}/eval.log"
