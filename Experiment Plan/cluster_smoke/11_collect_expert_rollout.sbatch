#!/usr/bin/env bash
#SBATCH --job-name=shadow-rollout
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=xgpf
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -Eeuo pipefail

: "${ISAACLAB_DIR:?Submit with ISAACLAB_DIR and VENV_DIR}"
: "${VENV_DIR:?Submit with ISAACLAB_DIR and VENV_DIR}"

source "${VENV_DIR}/bin/activate"
cd "${ISAACLAB_DIR}"

export PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA=YES
export XDG_CACHE_HOME="${HOME}/data/low-dim-intent/cache/xdg"

# ROLLOUT_TASK=m2 collects on the M2 task (shadow_zaxis_env.py); pass ROLLOUT_CHECKPOINT to the M2 expert.
case "${ROLLOUT_TASK:-stage1}" in
    stage1) task=Isaac-Repose-Cube-Shadow-Direct-v0; run_prefix=expert-rollout ;;
    m2) task=Isaac-Repose-Cube-Shadow-ZAxis-Direct-v0; run_prefix=expert-m2-rollout ;;
    faceturn) task=Isaac-FaceTurn-HingedCube-Shadow-Direct-v0; run_prefix=expert-faceturn-rollout
        : "${ROLLOUT_CHECKPOINT:?ROLLOUT_TASK=faceturn needs ROLLOUT_CHECKPOINT (a face-turn expert)}"
        # Face-turn task (hinged_cube_env.py) as its expert was trained and evaluated: torque cap 0.05 N m,
        # a turn counts after 10 held steps, flat start only.
        export HINGE_REFERENCE_JSON="${HINGE_REFERENCE_JSON:-${HOME}/data/low-dim-intent/runs/hinged-cube-eval-job1141499/dexcube.json}"
        export HINGE_EFFORT_LIMIT=0.05 FACE_TURN_HOLD_STEPS=10
        unset FACE_TURN_START_STATES ;;
    unified) task=Isaac-Unified-HingedCube-Shadow-Direct-v0; run_prefix=expert-unified-v4-rollout
        # E3: frozen v4 unified teacher on its own training env (unified_env.sh), as its students train.
        source "${SLURM_SUBMIT_DIR}/unified_env.sh"
        ROLLOUT_CHECKPOINT="${ROLLOUT_CHECKPOINT:-${HOME}/data/low-dim-intent/checkpoints/${unified_teacher_name}/frozen/${unified_teacher_file}}" ;;
    *) echo "unknown ROLLOUT_TASK=${ROLLOUT_TASK}" >&2; exit 2 ;;
esac
checkpoint="${ROLLOUT_CHECKPOINT:-${HOME}/data/low-dim-intent/checkpoints/expert-seed0/frozen/expert-seed0.pth}"
num_envs="${ROLLOUT_NUM_ENVS:-256}"
vector_steps="${ROLLOUT_VECTOR_STEPS:-700}"
seed="${ROLLOUT_SEED:-20261007}"
dataset_dir="${HOME}/data/low-dim-intent/datasets/${run_prefix}-seed0-job${SLURM_JOB_ID}"
run_root="${HOME}/data/low-dim-intent/runs/${run_prefix}-seed0-job${SLURM_JOB_ID}"
metadata_dir="${HOME}/data/low-dim-intent/metadata/${run_prefix}-seed0-job${SLURM_JOB_ID}"
collector="${SLURM_SUBMIT_DIR}/collect_shadow_rollout.py"

test -f "${checkpoint}"
test -f "${collector}"
mkdir -p "${run_root}" "${metadata_dir}" "${XDG_CACHE_HOME}"
cp "${collector}" "${metadata_dir}/collect_shadow_rollout.py"
cp "${SLURM_SUBMIT_DIR}"/{m2_tasks.py,shadow_zaxis_env.py,hinged_cube_tasks.py,hinged_cube_env.py} "${metadata_dir}/"
if [[ "${ROLLOUT_TASK:-stage1}" == faceturn || "${ROLLOUT_TASK:-stage1}" == unified ]]; then
    export HINGE_ASSET_DIR="${run_root}/asset"
    echo "hinge_reference_json=${HINGE_REFERENCE_JSON}"
    echo "hinge_effort_limit=${HINGE_EFFORT_LIMIT}"
    echo "face_turn_hold_steps=${FACE_TURN_HOLD_STEPS}"
    if [[ "${ROLLOUT_TASK}" == unified ]]; then
        cp "${SLURM_SUBMIT_DIR}/unified_env.sh" "${metadata_dir}/"
        env | grep -E '^UNIFIED_' | sort
    fi
fi
cp "${SLURM_SUBMIT_DIR}/11_collect_expert_rollout.sbatch" "${metadata_dir}/11_collect_expert_rollout.sbatch"
scontrol show job "${SLURM_JOB_ID}" > "${metadata_dir}/slurm-job.txt"

echo "record_type=shadow_expert_rollout_job"
echo "timestamp=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname -f)"
echo "partition=${SLURM_JOB_PARTITION}"
echo "isaaclab_commit=$(git rev-parse HEAD)"
echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd ';')"
echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | paste -sd ';')"
echo "task=${task}"
echo "checkpoint=${checkpoint}"
echo "checkpoint_sha256=$(sha256sum "${checkpoint}" | awk '{print $1}')"
echo "num_envs=${num_envs}"
echo "vector_steps=${vector_steps}"
echo "expected_samples=$((num_envs * vector_steps))"
echo "seed=${seed}"
echo "dataset_dir=${dataset_dir}"

run_log="${run_root}/collect.log"
gpu_log="${run_root}/gpu.csv"
nvidia-smi \
    --query-gpu=timestamp,name,memory.used,utilization.gpu \
    --format=csv,noheader,nounits \
    --loop=5 > "${gpu_log}" &
monitor_pid=$!
trap 'kill "${monitor_pid}" 2>/dev/null || true; wait "${monitor_pid}" 2>/dev/null || true' EXIT

start_epoch=$(date +%s)
set +e
/usr/bin/time -p timeout --signal=INT --kill-after=60s 1500s \
    ./isaaclab.sh -p "${collector}" \
    --task "${task}" \
    --checkpoint "${checkpoint}" \
    --output_dir "${dataset_dir}" \
    --num_envs "${num_envs}" \
    --vector_steps "${vector_steps}" \
    --seed "${seed}" \
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
echo "run_log=${run_log}"
echo "gpu_log=${gpu_log}"
awk -F, '
    {gsub(/ MiB/, "", $3); gsub(/ /, "", $3); if (($3 + 0) > max) max = $3 + 0}
    END {printf "peak_gpu_memory_mib=%d\n", max}
' "${gpu_log}"
cp "${run_log}" "${metadata_dir}/collect.log"
cp "${gpu_log}" "${metadata_dir}/gpu.csv"

if [[ "${command_status}" -eq 0 && -f "${dataset_dir}/manifest.json" ]]; then
    du -sh "${dataset_dir}"
    echo "rollout_job_status=PASS"
else
    echo "rollout_job_status=FAIL"
    exit "${command_status:-1}"
fi
