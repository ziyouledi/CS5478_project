#!/usr/bin/env bash
#SBATCH --job-name=shadow-student
#SBATCH --partition=gpu,gpu-long
#SBATCH --gpus=1
#SBATCH --constraint=xgpf
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# One student run per array task. Task index i maps to
#   intent = STUDENT_INTENTS[i / num_seeds], seed = STUDENT_SEEDS[i % num_seeds].
# Submit the stage-one grid (7 intents x 3 seeds) with --array=0-20%<max concurrent>.

set -Eeuo pipefail

: "${ISAACLAB_DIR:?Submit with ISAACLAB_DIR and VENV_DIR}"
: "${VENV_DIR:?Submit with ISAACLAB_DIR and VENV_DIR}"
: "${INTENT_PCA_DIR:?Submit with INTENT_PCA_DIR (a completed action-pca output directory)}"
: "${STUDENT_ITERATIONS:?Submit with STUDENT_ITERATIONS (PPO iterations per run)}"

source "${VENV_DIR}/bin/activate"
cd "${ISAACLAB_DIR}"

export PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA=YES
export XDG_CACHE_HOME="${HOME}/data/low-dim-intent/cache/xdg"

read -r -a intents <<< "${STUDENT_INTENTS:-blind wrist-2 hand-2 hand-4 hand-20 obj-1 obj-4}"
read -r -a seeds <<< "${STUDENT_SEEDS:-0 1 2}"
stage="${STUDENT_STAGE:-stage1}"
obj_lookahead="${OBJ_LOOKAHEAD:-0.5}"
num_envs=2048
task_id="${SLURM_ARRAY_TASK_ID:-0}"
if (( task_id >= ${#intents[@]} * ${#seeds[@]} )); then
    echo "array index ${task_id} is outside ${#intents[@]} intents x ${#seeds[@]} seeds" >&2
    exit 2
fi
intent="${intents[$((task_id / ${#seeds[@]}))]}"
seed="${seeds[$((task_id % ${#seeds[@]}))]}"

# STUDENT_STAGE=m2 trains on the M2 task (shadow_zaxis_env.py) with the frozen M2 expert.
case "${stage}" in
    m2) task=Isaac-Repose-Cube-Shadow-ZAxis-Direct-v0; expert_name=expert-m2-seed0 ;;
    *) task=Isaac-Repose-Cube-Shadow-Direct-v0; expert_name=expert-seed0 ;;
esac
frozen_dir="${HOME}/data/low-dim-intent/checkpoints/${expert_name}/frozen"
checkpoint="${frozen_dir}/${expert_name}.pth"
expert_sha256=$(awk -v name="${expert_name}.pth" '$2 == name {print $1}' "${frozen_dir}/SHA256SUMS")
# joint-K intents use the joint-position basis (20_fit_joint_basis.sbatch); all others the action PCA.
case "${intent}" in
    joint-*) pca_basis="${JOINT_PCA_DIR:?joint-K intents need JOINT_PCA_DIR}/joint_basis.npz" ;;
    *) pca_basis="${INTENT_PCA_DIR}/pca_basis.npz" ;;
esac
job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${task_id}"
name="student-${stage}-${intent}-seed${seed}"
run_root="${HOME}/data/low-dim-intent/runs/${name}-job${job_tag}"
metadata_dir="${HOME}/data/low-dim-intent/metadata/${name}-job${job_tag}"
trainer="${SLURM_SUBMIT_DIR}/train_student.py"

test -f "${checkpoint}"
test -n "${expert_sha256}"
test -f "${pca_basis}"
test -f "${trainer}"
mkdir -p "${run_root}" "${metadata_dir}" "${XDG_CACHE_HOME}"
cp "${SLURM_SUBMIT_DIR}"/{train_student.py,intent_modules.py,m2_tasks.py,shadow_zaxis_env.py,16_train_student.sbatch} "${metadata_dir}/"
scontrol show job "${SLURM_JOB_ID}" > "${metadata_dir}/slurm-job.txt"

echo "record_type=shadow_student_train_job"
echo "timestamp=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID}"
echo "array_job_id=${SLURM_ARRAY_JOB_ID:-none}"
echo "array_task_id=${task_id}"
echo "node=$(hostname -f)"
echo "isaaclab_commit=$(git rev-parse HEAD)"
echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd ';')"
echo "stage=${stage}"
echo "task=${task}"
echo "expert_checkpoint=${checkpoint}"
echo "intent=${intent}"
echo "seed=${seed}"
echo "num_envs=${num_envs}"
echo "iterations=${STUDENT_ITERATIONS}"
echo "expert_sha256=${expert_sha256}"
echo "pca_basis=${pca_basis}"
echo "obj_lookahead=${obj_lookahead}"
echo "run_root=${run_root}"

run_log="${run_root}/train.log"
gpu_log="${run_root}/gpu.csv"
nvidia-smi \
    --query-gpu=timestamp,name,memory.used,utilization.gpu \
    --format=csv,noheader,nounits \
    --loop=30 > "${gpu_log}" &
monitor_pid=$!
trap 'kill "${monitor_pid}" 2>/dev/null || true; wait "${monitor_pid}" 2>/dev/null || true' EXIT

start_epoch=$(date +%s)
set +e
/usr/bin/time -p timeout --signal=INT --kill-after=60s 10500s \
    ./isaaclab.sh -p "${trainer}" \
    --task "${task}" \
    --intent "${intent}" \
    --expert_checkpoint "${checkpoint}" \
    --expert_sha256 "${expert_sha256}" \
    --pca_basis "${pca_basis}" \
    --obj_lookahead "${obj_lookahead}" \
    --num_envs "${num_envs}" \
    --seed "${seed}" \
    --max_iterations "${STUDENT_ITERATIONS}" \
    --train_dir "${run_root}" \
    --experiment_name "${name}" \
    --headless \
    2>&1 | tee "${run_log}"
command_status=${PIPESTATUS[0]}
set -e
end_epoch=$(date +%s)

kill "${monitor_pid}" 2>/dev/null || true
wait "${monitor_pid}" 2>/dev/null || true
trap - EXIT

echo "elapsed_seconds=$((end_epoch - start_epoch))"
echo "command_status=${command_status}"
awk -F, '
    {gsub(/ /, "", $3); if (($3 + 0) > max) max = $3 + 0}
    END {printf "peak_gpu_memory_mib=%d\n", max}
' "${gpu_log}"
python - "${run_log}" <<'PY'
import re
import statistics
import sys

text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
fps = [int(value) for value in re.findall(r"fps total:\s*(\d+)", text)]
steady = fps[5:] if len(fps) > 5 else fps
print(f"reported_iteration_count={len(fps)}")
if steady:
    print(f"steady_total_fps_median={statistics.median(steady):.1f}")
checkpoints = re.findall(r"saving checkpoint '([^']+)'", text)
print(f"last_checkpoint={checkpoints[-1] if checkpoints else 'not_reported'}")
PY
cp "${run_log}" "${metadata_dir}/train.log"
cp "${gpu_log}" "${metadata_dir}/gpu.csv"

if [[ "${command_status}" -eq 0 ]] && grep -q "student_train_status=PASS" "${run_log}"; then
    echo "student_job_status=PASS"
else
    echo "student_job_status=FAIL"
    exit "${command_status:-1}"
fi
