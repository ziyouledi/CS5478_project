#!/usr/bin/env bash
#SBATCH --job-name=shadow-smoke
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=xgpf
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -Eeuo pipefail

: "${ISAACLAB_DIR:?Submit with --export=ALL,ISAACLAB_DIR=/absolute/path/to/IsaacLab,VENV_DIR=/absolute/path/to/env_isaaclab}"
: "${VENV_DIR:?Submit with --export=ALL,ISAACLAB_DIR=/absolute/path/to/IsaacLab,VENV_DIR=/absolute/path/to/env_isaaclab}"

source "${VENV_DIR}/bin/activate"
cd "${ISAACLAB_DIR}"

export PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA=YES
export XDG_CACHE_HOME="${HOME}/data/low-dim-intent/cache/xdg"
mkdir -p "${XDG_CACHE_HOME}" "${ISAACLAB_DIR}/logs/smoke"

echo "record_type=shadow_random_agent_smoke"
echo "timestamp=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname -f)"
echo "partition=${SLURM_JOB_PARTITION}"
echo "isaaclab_dir=${ISAACLAB_DIR}"
echo "isaaclab_commit=$(git rev-parse HEAD)"
echo "python=$(python --version 2>&1)"
echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd ';')"
echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | paste -sd ';')"
echo "torch_cuda_available=$(python -c 'import torch; print(torch.cuda.is_available())')"
echo "torch_cuda_build=$(python -c 'import torch; print(torch.version.cuda)')"

start_epoch=$(date +%s)
run_log="${ISAACLAB_DIR}/logs/smoke/shadow-smoke-${SLURM_JOB_ID}.log"
gpu_log="${ISAACLAB_DIR}/logs/smoke/shadow-smoke-${SLURM_JOB_ID}-gpu.csv"

nvidia-smi \
    --query-gpu=timestamp,name,memory.used,utilization.gpu \
    --format=csv,noheader,nounits \
    --loop=2 > "${gpu_log}" &
monitor_pid=$!

set +e
/usr/bin/time -p timeout --signal=INT --kill-after=30s 1500s \
    ./isaaclab.sh -p "${SLURM_SUBMIT_DIR}/shadow_smoke.py" \
    --task Isaac-Repose-Cube-Shadow-Direct-v0 \
    --num_envs 64 \
    --steps 200 \
    --seed 0 \
    --headless 2>&1 | tee "${run_log}"
command_status=$?
kill "${monitor_pid}" 2>/dev/null || true
wait "${monitor_pid}" 2>/dev/null || true
set -e
end_epoch=$(date +%s)

echo "elapsed_seconds=$((end_epoch - start_epoch))"
echo "command_status=${command_status}"
echo "run_log=${run_log}"
echo "gpu_log=${gpu_log}"
awk -F, '
    {gsub(/ MiB/, "", $3); gsub(/ /, "", $3); if (($3 + 0) > max) max = $3 + 0}
    END {printf "peak_gpu_memory_mib=%d\n", max}
' "${gpu_log}"

if [[ "${command_status}" -eq 0 ]] \
    && grep -Fq "[INFO]: Completed setting up the environment" "${run_log}" \
    && grep -Fq "completed_step=200" "${run_log}" \
    && grep -Fq "python_smoke_status=PASS" "${run_log}"; then
    echo "smoke_status=PASS"
else
    echo "smoke_status=FAIL"
    if [[ "${command_status}" -eq 0 ]]; then
        exit 1
    fi
    exit "${command_status}"
fi
