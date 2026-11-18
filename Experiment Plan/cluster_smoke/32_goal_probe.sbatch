#!/usr/bin/env bash
#SBATCH --job-name=shadow-goal-probe
#SBATCH --partition=normal
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Offline goal probe (analyze_goal_probe.py): goal information in a held hand-8 code, beyond the goal-hidden state.
# Array task 0 = face-turn expert rollout (job1153675, PCA job1153676), task 1 = Stage 1 / M1 expert rollout
# (job1023537, PCA job1024096). Submit with --array=0-1 and VENV_DIR.

set -Eeuo pipefail

: "${VENV_DIR:?Submit with VENV_DIR}"
source "${VENV_DIR}/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

data="${HOME}/data/low-dim-intent/datasets"
task_id="${SLURM_ARRAY_TASK_ID:-0}"
case "${task_id}" in
    0) dataset_dir="${data}/expert-faceturn-rollout-seed0-job1153675"
       pca_dir="${data}/action-pca-expert-faceturn-rollout-seed0-job1153675-job1153676"
       extra=(--face_turn) ;;
    1) dataset_dir="${data}/expert-rollout-seed0-job1023537"
       pca_dir="${data}/action-pca-expert-rollout-seed0-job1023537-job1024096"
       extra=() ;;
    *) echo "unknown array index ${task_id}" >&2; exit 2 ;;
esac
holds="${PROBE_HOLDS:-1,3,6,12,30}"
job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${task_id}"
output_dir="${data}/goal-probe-$(basename "${dataset_dir}")-job${job_tag}"
metadata_dir="${HOME}/data/low-dim-intent/metadata/goal-probe-job${job_tag}"
analyzer="${SLURM_SUBMIT_DIR}/analyze_goal_probe.py"

test -f "${dataset_dir}/manifest.json"
test -f "${pca_dir}/pca_basis.npz"
test -f "${analyzer}"
mkdir -p "${metadata_dir}"
cp "${SLURM_SUBMIT_DIR}"/{analyze_goal_probe.py,analyze_intent_hold.py,intent_modules.py,32_goal_probe.sbatch} "${metadata_dir}/"
scontrol show job "${SLURM_JOB_ID}" > "${metadata_dir}/slurm-job.txt"

echo "record_type=shadow_goal_probe_job"
echo "timestamp=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID} array_task_id=${task_id}"
echo "node=$(hostname -f)"
echo "dataset_dir=${dataset_dir}"
echo "pca_dir=${pca_dir}"
echo "pca_basis_check=$(cd "${pca_dir}" && sha256sum -c --strict SHA256SUMS >/dev/null && echo PASS)"
echo "holds=${holds}"
echo "output_dir=${output_dir}"

echo "dataset_checksum_verification=START"
(cd "${dataset_dir}" && sha256sum -c --strict SHA256SUMS)
echo "dataset_checksum_verification=PASS"

cd "${SLURM_SUBMIT_DIR}"
start_epoch=$(date +%s)
set +e
python "${analyzer}" --dataset_dir "${dataset_dir}" --pca_basis "${pca_dir}/pca_basis.npz" "${extra[@]}" \
    --holds "${holds}" --output_dir "${output_dir}" 2>&1 | tee "${metadata_dir}/goal_probe.log"
command_status=${PIPESTATUS[0]}
set -e
echo "elapsed_seconds=$(( $(date +%s) - start_epoch ))"
echo "command_status=${command_status}"
if [[ "${command_status}" -eq 0 ]] && grep -q "probe_status=PASS" "${metadata_dir}/goal_probe.log"; then
    echo "goal_probe_job_status=PASS"
else
    echo "goal_probe_job_status=FAIL"
    exit "${command_status:-1}"
fi
