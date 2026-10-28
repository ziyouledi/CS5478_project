#!/usr/bin/env bash
#SBATCH --job-name=shadow-student-eval
#SBATCH --partition=gpu,gpu-long
#SBATCH --gpus=1
#SBATCH --constraint=xgpf
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# One run per array task. EVAL_RUN_DIRS is a space-separated list of experiment directories
# (each containing nn/ and, for students, params/intent.json); task i evaluates entry i.
# EVAL_INTENT is only needed for runs without params/intent.json (e.g. EVAL_INTENT=blind for the expert).
# EVAL_TASK selects the gym task (default: Stage 1; Isaac-Repose-Cube-Shadow-ZAxis-Direct-v0 for M2).

set -Eeuo pipefail

: "${ISAACLAB_DIR:?Submit with ISAACLAB_DIR and VENV_DIR}"
: "${VENV_DIR:?Submit with ISAACLAB_DIR and VENV_DIR}"
: "${EVAL_RUN_DIRS:?Submit with EVAL_RUN_DIRS (space-separated experiment directories)}"

source "${VENV_DIR}/bin/activate"
cd "${ISAACLAB_DIR}"

export PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA=YES
export XDG_CACHE_HOME="${HOME}/data/low-dim-intent/cache/xdg"

read -r -a run_dirs <<< "${EVAL_RUN_DIRS}"
task_id="${SLURM_ARRAY_TASK_ID:-0}"
if (( task_id >= ${#run_dirs[@]} )); then
    echo "array index ${task_id} is outside ${#run_dirs[@]} run directories" >&2
    exit 2
fi
run_dir="${run_dirs[${task_id}]}"
stride="${EVAL_CHECKPOINT_STRIDE:-1}"
num_envs="${EVAL_NUM_ENVS:-1024}"
seed="${EVAL_SEED:-20261028}"
task="${EVAL_TASK:-Isaac-Repose-Cube-Shadow-Direct-v0}"
job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${task_id}"
output_dir="${HOME}/data/low-dim-intent/runs/eval-$(basename "${run_dir}")-job${job_tag}"
metadata_dir="${HOME}/data/low-dim-intent/metadata/eval-$(basename "${run_dir}")-job${job_tag}"
evaluator="${SLURM_SUBMIT_DIR}/evaluate_students.py"

test -d "${run_dir}/nn"
test -f "${evaluator}"
mkdir -p "${output_dir}" "${metadata_dir}" "${XDG_CACHE_HOME}"
cp "${SLURM_SUBMIT_DIR}"/{evaluate_students.py,student_eval_utils.py,intent_env_wrapper.py,intent_modules.py,m2_tasks.py,shadow_zaxis_env.py,17_evaluate_students.sbatch} "${metadata_dir}/"
scontrol show job "${SLURM_JOB_ID}" > "${metadata_dir}/slurm-job.txt"

echo "record_type=shadow_student_evaluation_job"
echo "timestamp=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID}"
echo "array_task_id=${task_id}"
echo "node=$(hostname -f)"
echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd ';')"
echo "run_dir=${run_dir}"
echo "task=${task}"
echo "intent_override=${EVAL_INTENT:-none}"
echo "checkpoint_stride=${stride}"
echo "num_envs=${num_envs}"
echo "seed=${seed}"
echo "output_dir=${output_dir}"

eval_args=(
    --task "${task}"
    --run_dir "${run_dir}"
    --checkpoint_stride "${stride}"
    --num_envs "${num_envs}"
    --seed "${seed}"
    --output_dir "${output_dir}"
    --headless
)
if [[ -n "${EVAL_INTENT:-}" ]]; then
    eval_args+=(--intent "${EVAL_INTENT}")
fi

start_epoch=$(date +%s)
set +e
timeout --signal=INT --kill-after=30s 5100s \
    ./isaaclab.sh -p "${evaluator}" "${eval_args[@]}" 2>&1 | tee "${output_dir}/evaluate.log"
command_status=${PIPESTATUS[0]}
set -e
echo "elapsed_seconds=$(( $(date +%s) - start_epoch ))"
echo "command_status=${command_status}"
cp "${output_dir}/evaluate.log" "${metadata_dir}/"

if [[ "${command_status}" -eq 0 ]] && grep -q "student_evaluation_status=PASS" "${output_dir}/evaluate.log"; then
    cp "${output_dir}/curve.csv" "${metadata_dir}/"
    echo "student_eval_job_status=PASS"
else
    echo "student_eval_job_status=FAIL"
    exit "${command_status:-1}"
fi
