#!/usr/bin/env bash
#SBATCH --job-name=shadow-intent-check
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -Eeuo pipefail

: "${VENV_DIR:?Submit with VENV_DIR}"
: "${INTENT_PCA_DIR:?Submit with INTENT_PCA_DIR (a completed action-pca output directory)}"
: "${OBJ_LOOKAHEAD:?Submit with OBJ_LOOKAHEAD (radians)}"

source "${VENV_DIR}/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

dataset_dir="${INTENT_DATASET_DIR:-${HOME}/data/low-dim-intent/datasets/expert-rollout-seed0-job1023537}"
frozen_dir="${HOME}/data/low-dim-intent/checkpoints/expert-seed0/frozen"
checkpoint="${frozen_dir}/expert-seed0.pth"
expert_sha256=$(awk '$2 == "expert-seed0.pth" {print $1}' "${frozen_dir}/SHA256SUMS")
pca_basis="${INTENT_PCA_DIR}/pca_basis.npz"
metadata_dir="${HOME}/data/low-dim-intent/metadata/intent-check-job${SLURM_JOB_ID}"

test -f "${checkpoint}"
test -n "${expert_sha256}"
test -f "${pca_basis}"
mkdir -p "${metadata_dir}"
cp "${SLURM_SUBMIT_DIR}"/{intent_modules.py,validate_intent_modules.py,14_validate_intents.sbatch} "${metadata_dir}/"
scontrol show job "${SLURM_JOB_ID}" > "${metadata_dir}/slurm-job.txt"

echo "record_type=shadow_intent_validation_job"
echo "timestamp=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname -f)"
echo "torch=$(python -c 'import torch; print(torch.__version__)')"
echo "dataset_dir=${dataset_dir}"
echo "checkpoint=${checkpoint}"
echo "expert_sha256=${expert_sha256}"
echo "pca_basis=${pca_basis}"
echo "pca_basis_check=$(cd "${INTENT_PCA_DIR}" && sha256sum -c --strict SHA256SUMS >/dev/null && echo PASS)"
echo "obj_lookahead=${OBJ_LOOKAHEAD}"

set +e
python "${SLURM_SUBMIT_DIR}/validate_intent_modules.py" \
    --dataset_dir "${dataset_dir}" \
    --pca_basis "${pca_basis}" \
    --expert_checkpoint "${checkpoint}" \
    --expert_sha256 "${expert_sha256}" \
    --obj_lookahead "${OBJ_LOOKAHEAD}" \
    --output_json "${metadata_dir}/intent_validation.json" \
    2>&1 | tee "${metadata_dir}/validate.log"
command_status=${PIPESTATUS[0]}
set -e

echo "command_status=${command_status}"
if [[ "${command_status}" -eq 0 ]]; then
    echo "intent_check_job_status=PASS"
else
    echo "intent_check_job_status=FAIL"
    exit "${command_status}"
fi
