#!/usr/bin/env bash
#SBATCH --job-name=shadow-joint-pca
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Diagnostic: PCA of the expert's measured joint positions vs its action commands (same rollout).

set -Eeuo pipefail

: "${VENV_DIR:?Submit with VENV_DIR}"
: "${PCA_DATASET_DIR:?Submit with PCA_DATASET_DIR (a completed expert rollout directory)}"

source "${VENV_DIR}/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="${HOME}/data/low-dim-intent/cache/matplotlib"

dataset_dir="${PCA_DATASET_DIR}"
output_dir="${HOME}/data/low-dim-intent/datasets/joint-pca-$(basename "${dataset_dir}")-job${SLURM_JOB_ID}"
metadata_dir="${HOME}/data/low-dim-intent/metadata/joint-pca-job${SLURM_JOB_ID}"

test -f "${dataset_dir}/manifest.json"
mkdir -p "${metadata_dir}" "${MPLCONFIGDIR}"
cp "${SLURM_SUBMIT_DIR}"/{analyze_joint_pca.py,analyze_action_pca.py,18_joint_pca.sbatch} "${metadata_dir}/"
scontrol show job "${SLURM_JOB_ID}" > "${metadata_dir}/slurm-job.txt"

echo "record_type=shadow_joint_pca_job"
echo "timestamp=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname -f)"
echo "dataset_dir=${dataset_dir}"
echo "output_dir=${output_dir}"

echo "dataset_checksum_verification=START"
(cd "${dataset_dir}" && sha256sum -c --strict SHA256SUMS)
echo "dataset_checksum_verification=PASS"

set +e
python "${SLURM_SUBMIT_DIR}/analyze_joint_pca.py" --dataset_dir "${dataset_dir}" --output_dir "${output_dir}" \
    2>&1 | tee "${metadata_dir}/joint_pca.log"
command_status=${PIPESTATUS[0]}
set -e
echo "command_status=${command_status}"

if [[ "${command_status}" -eq 0 && -f "${output_dir}/summary.json" ]]; then
    cp "${output_dir}/summary.json" "${output_dir}/cumulative_variance.csv" "${metadata_dir}/"
    echo "joint_pca_job_status=PASS"
else
    echo "joint_pca_job_status=FAIL"
    exit "${command_status:-1}"
fi
