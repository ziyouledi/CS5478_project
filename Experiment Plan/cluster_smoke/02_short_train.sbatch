#!/usr/bin/env bash
#SBATCH --job-name=shadow-ppo-short
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=xgpf
#SBATCH --time=00:30:00
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
mkdir -p "${XDG_CACHE_HOME}" "${ISAACLAB_DIR}/logs/ppo-short"

echo "record_type=shadow_ppo_short"
echo "timestamp=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname -f)"
echo "partition=${SLURM_JOB_PARTITION}"
echo "isaaclab_commit=$(git rev-parse HEAD)"
echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd ';')"
echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | paste -sd ';')"
echo "torch_cuda_available=$(python -c 'import torch; print(torch.cuda.is_available())')"
echo "torch_cuda_build=$(python -c 'import torch; print(torch.version.cuda)')"

agent_cfg_file="${ISAACLAB_DIR}/source/isaaclab_tasks/isaaclab_tasks/direct/shadow_hand/agents/rl_games_ppo_cfg.yaml"
read -r horizon_length default_minibatch rollout_batch adjusted_minibatch < <(
    python - "${agent_cfg_file}" <<'PY'
import math
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as stream:
    cfg = yaml.safe_load(stream)
train_cfg = cfg["params"]["config"]
horizon = int(train_cfg["horizon_length"])
minibatch = int(train_cfg["minibatch_size"])
batch = 512 * horizon
print(horizon, minibatch, batch, math.gcd(batch, minibatch))
PY
)
echo "horizon_length=${horizon_length}"
echo "default_minibatch_size=${default_minibatch}"
echo "rollout_batch_size=${rollout_batch}"
echo "short_run_minibatch_size=${adjusted_minibatch}"

start_epoch=$(date +%s)
run_log="${ISAACLAB_DIR}/logs/ppo-short/shadow-ppo-short-${SLURM_JOB_ID}.log"
gpu_log="${ISAACLAB_DIR}/logs/ppo-short/shadow-ppo-short-${SLURM_JOB_ID}-gpu.csv"

nvidia-smi \
    --query-gpu=timestamp,name,memory.used,utilization.gpu \
    --format=csv,noheader,nounits \
    --loop=2 > "${gpu_log}" &
monitor_pid=$!

set +e
/usr/bin/time -p timeout --signal=INT --kill-after=30s 1500s \
    ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Repose-Cube-Shadow-Direct-v0 \
    --num_envs 512 \
    --max_iterations 5 \
    --seed 0 \
    --headless \
    agent.params.config.minibatch_size="${adjusted_minibatch}" \
    2>&1 | tee "${run_log}"
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

if [[ "${command_status}" -eq 0 ]]; then
    echo "short_train_status=PASS"
else
    echo "short_train_status=FAIL"
    exit "${command_status}"
fi
