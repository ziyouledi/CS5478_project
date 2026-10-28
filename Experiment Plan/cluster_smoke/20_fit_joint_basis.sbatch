#!/usr/bin/env bash
#SBATCH --job-name=shadow-joint-basis
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Fit the joint-position PCA basis of the joint-K intent (fit_joint_basis.py) on one expert rollout.

set -Eeuo pipefail

: "${VENV_DIR:?Submit with VENV_DIR}"
: "${BASIS_DATASET_DIR:?Submit with BASIS_DATASET_DIR (a completed expert rollout directory)}"

source "${VENV_DIR}/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

dataset_dir="${BASIS_DATASET_DIR}"
action_pca_dir="${BASIS_ACTION_PCA_DIR:-}"
output_dir="${HOME}/data/low-dim-intent/datasets/joint-basis-$(basename "${dataset_dir}")-job${SLURM_JOB_ID}"
metadata_dir="${HOME}/data/low-dim-intent/metadata/joint-basis-job${SLURM_JOB_ID}"

test -f "${dataset_dir}/manifest.json"
mkdir -p "${metadata_dir}"
cp "${SLURM_SUBMIT_DIR}"/{fit_joint_basis.py,analyze_action_pca.py,20_fit_joint_basis.sbatch} "${metadata_dir}/"
scontrol show job "${SLURM_JOB_ID}" > "${metadata_dir}/slurm-job.txt"

echo "record_type=shadow_joint_basis_job"
echo "timestamp=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname -f)"
echo "dataset_dir=${dataset_dir}"
echo "action_pca_dir=${action_pca_dir:-none}"
echo "output_dir=${output_dir}"

echo "dataset_checksum_verification=START"
(cd "${dataset_dir}" && sha256sum -c --strict SHA256SUMS)
echo "dataset_checksum_verification=PASS"

fit_args=(--dataset_dir "${dataset_dir}" --output_dir "${output_dir}")
if [[ -n "${action_pca_dir}" ]]; then
    fit_args+=(--action_pca_basis "${action_pca_dir}/pca_basis.npz")
fi

set +e
python "${SLURM_SUBMIT_DIR}/fit_joint_basis.py" "${fit_args[@]}" 2>&1 | tee "${metadata_dir}/fit.log"
command_status=${PIPESTATUS[0]}
set -e
echo "command_status=${command_status}"

if [[ "${command_status}" -eq 0 && -f "${output_dir}/joint_basis.npz" ]]; then
    (cd "${output_dir}" && sha256sum joint_basis.npz summary.json > SHA256SUMS)
    cp "${output_dir}"/{summary.json,SHA256SUMS} "${metadata_dir}/"
    echo "joint_basis_job_status=PASS"
else
    echo "joint_basis_job_status=FAIL"
    exit "${command_status:-1}"
fi
