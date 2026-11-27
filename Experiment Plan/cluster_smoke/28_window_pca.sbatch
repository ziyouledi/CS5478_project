#!/usr/bin/env bash
#SBATCH --job-name=shadow-window-pca
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -Eeuo pipefail

: "${VENV_DIR:?Submit with VENV_DIR}"

source "${VENV_DIR}/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="${HOME}/data/low-dim-intent/cache/matplotlib"

dataset_dir="${HOLD_DATASET_DIR:-${HOME}/data/low-dim-intent/datasets/expert-rollout-seed0-job1023537}"
lengths="${WINDOW_LENGTHS:-1,3,6,12,21,30}"
output_dir="${HOME}/data/low-dim-intent/datasets/window-pca-$(basename "${dataset_dir}")-job${SLURM_JOB_ID}"
metadata_dir="${HOME}/data/low-dim-intent/metadata/window-pca-job${SLURM_JOB_ID}"
analyzer="${SLURM_SUBMIT_DIR}/analyze_window_pca.py"

test -f "${dataset_dir}/manifest.json"
test -f "${dataset_dir}/SHA256SUMS"
test -f "${analyzer}"
mkdir -p "${metadata_dir}" "${MPLCONFIGDIR}"
cp "${SLURM_SUBMIT_DIR}"/{analyze_window_pca.py,analyze_joint_timescales.py,analyze_pc_timescales.py,analyze_intent_hold.py,intent_modules.py,28_window_pca.sbatch} "${metadata_dir}/"
scontrol show job "${SLURM_JOB_ID}" > "${metadata_dir}/slurm-job.txt"

echo "record_type=shadow_window_pca_job"
echo "timestamp=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname -f)"
echo "partition=${SLURM_JOB_PARTITION}"
echo "torch=$(python -c 'import torch; print(torch.__version__)')"
echo "dataset_dir=${dataset_dir}"
echo "lengths=${lengths}"
echo "output_dir=${output_dir}"

echo "dataset_checksum_verification=START"
(cd "${dataset_dir}" && sha256sum -c --strict SHA256SUMS)
echo "dataset_checksum_verification=PASS"

start_epoch=$(date +%s)
set +e
python "${analyzer}" \
    --dataset_dir "${dataset_dir}" \
    --lengths "${lengths}" \
    --output_dir "${output_dir}" \
    2>&1 | tee "${metadata_dir}/window_pca.log"
command_status=${PIPESTATUS[0]}
set -e
end_epoch=$(date +%s)

echo "elapsed_seconds=$((end_epoch - start_epoch))"
echo "command_status=${command_status}"

if [[ "${command_status}" -eq 0 && -f "${output_dir}/summary.json" ]]; then
    cp "${output_dir}/summary.json" "${metadata_dir}/summary.json"
    ls -la "${output_dir}"
    echo "window_pca_job_status=PASS"
else
    echo "window_pca_job_status=FAIL"
    exit "${command_status:-1}"
fi
