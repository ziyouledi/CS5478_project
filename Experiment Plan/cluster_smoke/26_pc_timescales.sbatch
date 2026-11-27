#!/usr/bin/env bash
#SBATCH --job-name=shadow-pc-timescales
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -Eeuo pipefail

: "${VENV_DIR:?Submit with VENV_DIR}"
: "${INTENT_PCA_DIR:?Submit with INTENT_PCA_DIR (a completed action-pca output directory)}"

source "${VENV_DIR}/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="${HOME}/data/low-dim-intent/cache/matplotlib"

dataset_dir="${HOLD_DATASET_DIR:-${HOME}/data/low-dim-intent/datasets/expert-rollout-seed0-job1023537}"
holds="${HOLD_STEPS:-1,2,3,6,12,30}"
lags="${PC_LAGS:-1,2,3,4,6,8,12,20,30,60}"
pca_basis="${INTENT_PCA_DIR}/pca_basis.npz"
output_dir="${HOME}/data/low-dim-intent/datasets/pc-timescales-$(basename "${dataset_dir}")-job${SLURM_JOB_ID}"
metadata_dir="${HOME}/data/low-dim-intent/metadata/pc-timescales-job${SLURM_JOB_ID}"
analyzer="${SLURM_SUBMIT_DIR}/analyze_pc_timescales.py"

test -f "${dataset_dir}/manifest.json"
test -f "${dataset_dir}/SHA256SUMS"
test -f "${pca_basis}"
test -f "${analyzer}"
mkdir -p "${metadata_dir}" "${MPLCONFIGDIR}"
cp "${SLURM_SUBMIT_DIR}"/{analyze_pc_timescales.py,analyze_intent_hold.py,intent_modules.py,26_pc_timescales.sbatch} "${metadata_dir}/"
scontrol show job "${SLURM_JOB_ID}" > "${metadata_dir}/slurm-job.txt"

echo "record_type=shadow_pc_timescales_job"
echo "timestamp=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname -f)"
echo "partition=${SLURM_JOB_PARTITION}"
echo "torch=$(python -c 'import torch; print(torch.__version__)')"
echo "dataset_dir=${dataset_dir}"
echo "pca_basis=${pca_basis}"
echo "pca_basis_check=$(cd "${INTENT_PCA_DIR}" && sha256sum -c --strict SHA256SUMS >/dev/null && echo PASS)"
echo "lags=${lags}"
echo "holds=${holds}"
echo "output_dir=${output_dir}"

echo "dataset_checksum_verification=START"
(cd "${dataset_dir}" && sha256sum -c --strict SHA256SUMS)
echo "dataset_checksum_verification=PASS"

start_epoch=$(date +%s)
set +e
python "${analyzer}" \
    --dataset_dir "${dataset_dir}" \
    --pca_basis "${pca_basis}" \
    --lags "${lags}" \
    --holds "${holds}" \
    --output_dir "${output_dir}" \
    2>&1 | tee "${metadata_dir}/pc_timescales.log"
command_status=${PIPESTATUS[0]}
set -e
end_epoch=$(date +%s)

echo "elapsed_seconds=$((end_epoch - start_epoch))"
echo "command_status=${command_status}"

if [[ "${command_status}" -eq 0 && -f "${output_dir}/summary.json" ]]; then
    cp "${output_dir}/summary.json" "${metadata_dir}/summary.json"
    ls -la "${output_dir}"
    echo "pc_timescales_job_status=PASS"
else
    echo "pc_timescales_job_status=FAIL"
    exit "${command_status:-1}"
fi
