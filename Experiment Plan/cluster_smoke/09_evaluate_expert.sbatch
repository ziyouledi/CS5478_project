#!/usr/bin/env bash
#SBATCH --job-name=shadow-eval
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

checkpoint_dir="${CHECKPOINT_DIR:-${HOME}/data/low-dim-intent/checkpoints/expert-seed0}"
best_checkpoint="${BEST_CHECKPOINT:-${checkpoint_dir}/expert-seed0-best.pth}"
final_checkpoint="${FINAL_CHECKPOINT:-${checkpoint_dir}/expert-seed0-final.pth}"
num_envs="${EVAL_NUM_ENVS:-256}"
episodes="${EVAL_EPISODES:-1024}"
seed="${EVAL_SEED:-20261006}"
run_root="${HOME}/data/low-dim-intent/runs/expert-eval-job${SLURM_JOB_ID}"
metadata_dir="${HOME}/data/low-dim-intent/metadata/expert-eval-job${SLURM_JOB_ID}"
mkdir -p "${run_root}" "${metadata_dir}" "${XDG_CACHE_HOME}"

evaluator="${SLURM_SUBMIT_DIR}/evaluate_shadow_checkpoint.py"
test -f "${evaluator}"
test -f "${best_checkpoint}"
test -f "${final_checkpoint}"
cp "${evaluator}" "${metadata_dir}/evaluate_shadow_checkpoint.py"
cp "${SLURM_SUBMIT_DIR}/09_evaluate_expert.sbatch" "${metadata_dir}/09_evaluate_expert.sbatch"
scontrol show job "${SLURM_JOB_ID}" > "${metadata_dir}/slurm-job.txt"

echo "record_type=shadow_expert_checkpoint_comparison"
echo "timestamp=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname -f)"
echo "partition=${SLURM_JOB_PARTITION}"
echo "isaaclab_commit=$(git rev-parse HEAD)"
echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd ';')"
echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | paste -sd ';')"
echo "num_envs=${num_envs}"
echo "episodes_per_checkpoint=${episodes}"
echo "evaluation_seed=${seed}"
echo "best_checkpoint=${best_checkpoint}"
echo "final_checkpoint=${final_checkpoint}"
echo "run_root=${run_root}"

gpu_log="${run_root}/gpu.csv"
nvidia-smi \
    --query-gpu=timestamp,name,memory.used,utilization.gpu \
    --format=csv,noheader,nounits \
    --loop=5 > "${gpu_log}" &
monitor_pid=$!
trap 'kill "${monitor_pid}" 2>/dev/null || true; wait "${monitor_pid}" 2>/dev/null || true' EXIT

overall_status=0
for label in best final; do
    if [[ "${label}" == "best" ]]; then
        checkpoint="${best_checkpoint}"
    else
        checkpoint="${final_checkpoint}"
    fi
    result_json="${run_root}/${label}.json"
    run_log="${run_root}/${label}.log"
    echo "--- evaluation_start label=${label} ---"
    start_epoch=$(date +%s)
    set +e
    /usr/bin/time -p timeout --signal=INT --kill-after=60s 780s \
        ./isaaclab.sh -p "${evaluator}" \
        --task Isaac-Repose-Cube-Shadow-Direct-v0 \
        --checkpoint "${checkpoint}" \
        --num_envs "${num_envs}" \
        --episodes "${episodes}" \
        --seed "${seed}" \
        --output "${result_json}" \
        --headless \
        2>&1 | tee "${run_log}"
    command_status=${PIPESTATUS[0]}
    set -e
    end_epoch=$(date +%s)
    echo "evaluation_label=${label}"
    echo "evaluation_elapsed_seconds=$((end_epoch - start_epoch))"
    echo "evaluation_command_status=${command_status}"
    echo "evaluation_result=${result_json}"
    echo "evaluation_log=${run_log}"
    cp "${run_log}" "${metadata_dir}/${label}.log"
    if [[ -f "${result_json}" ]]; then
        cp "${result_json}" "${metadata_dir}/${label}.json"
    fi
    if [[ "${command_status}" -ne 0 ]]; then
        overall_status="${command_status}"
    fi
    echo "--- evaluation_end label=${label} ---"
done

kill "${monitor_pid}" 2>/dev/null || true
wait "${monitor_pid}" 2>/dev/null || true
trap - EXIT
cp "${gpu_log}" "${metadata_dir}/gpu.csv"
awk -F, '
    {gsub(/ MiB/, "", $3); gsub(/ /, "", $3); if (($3 + 0) > max) max = $3 + 0}
    END {printf "peak_gpu_memory_mib=%d\n", max}
' "${gpu_log}"

if [[ "${overall_status}" -eq 0 ]]; then
    echo "expert_evaluation_status=PASS"
else
    echo "expert_evaluation_status=FAIL"
    exit "${overall_status}"
fi
