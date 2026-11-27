#!/usr/bin/env bash
#SBATCH --job-name=shadow-action-pca
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -Eeuo pipefail

: "${VENV_DIR:?Submit with VENV_DIR}"
: "${PCA_DATASET_DIR:?Submit with PCA_DATASET_DIR (a completed expert rollout directory)}"

source "${VENV_DIR}/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="${HOME}/data/low-dim-intent/cache/matplotlib"

dataset_dir="${PCA_DATASET_DIR}"
expected_samples="${PCA_EXPECTED_SAMPLES:-1228800}"
clip_actions="${PCA_CLIP_ACTIONS:-}"
output_dir="${HOME}/data/low-dim-intent/datasets/action-pca-$(basename "${dataset_dir}")-job${SLURM_JOB_ID}"
metadata_dir="${HOME}/data/low-dim-intent/metadata/action-pca-job${SLURM_JOB_ID}"
analyzer="${SLURM_SUBMIT_DIR}/analyze_action_pca.py"

test -f "${dataset_dir}/manifest.json"
test -f "${dataset_dir}/SHA256SUMS"
test -f "${analyzer}"
mkdir -p "${metadata_dir}" "${MPLCONFIGDIR}"
cp "${analyzer}" "${metadata_dir}/analyze_action_pca.py"
cp "${SLURM_SUBMIT_DIR}/12_action_pca.sbatch" "${metadata_dir}/12_action_pca.sbatch"
scontrol show job "${SLURM_JOB_ID}" > "${metadata_dir}/slurm-job.txt"

echo "record_type=shadow_action_pca_job"
echo "timestamp=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname -f)"
echo "partition=${SLURM_JOB_PARTITION}"
echo "python=$(python --version 2>&1)"
echo "numpy=$(python -c 'import numpy; print(numpy.__version__)')"
echo "dataset_dir=${dataset_dir}"
echo "expected_samples=${expected_samples}"
echo "clip_actions=${clip_actions:-none}"
echo "output_dir=${output_dir}"

echo "dataset_checksum_verification=START"
(cd "${dataset_dir}" && sha256sum -c --strict SHA256SUMS)
echo "dataset_checksum_verification=PASS"

pca_args=(
    --dataset_dir "${dataset_dir}"
    --output_dir "${output_dir}"
    --expected_samples "${expected_samples}"
)
if [[ -n "${clip_actions}" ]]; then
    pca_args+=(--clip_actions "${clip_actions}")
fi

start_epoch=$(date +%s)
set +e
python "${analyzer}" "${pca_args[@]}" 2>&1 | tee "${metadata_dir}/pca.log"
command_status=${PIPESTATUS[0]}
set -e
end_epoch=$(date +%s)

echo "elapsed_seconds=$((end_epoch - start_epoch))"
echo "command_status=${command_status}"

if [[ "${command_status}" -eq 0 && -f "${output_dir}/summary.json" ]]; then
    cp "${output_dir}/summary.json" "${metadata_dir}/summary.json"
    ls -la "${output_dir}"
    echo "pca_job_status=PASS"
else
    echo "pca_job_status=FAIL"
    exit "${command_status:-1}"
fi
