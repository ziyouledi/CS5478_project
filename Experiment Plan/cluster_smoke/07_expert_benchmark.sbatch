#!/usr/bin/env bash
#SBATCH --job-name=shadow-expert-bench
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
mkdir -p "${XDG_CACHE_HOME}" "${ISAACLAB_DIR}/logs/expert-benchmark"

num_envs=2048
max_iterations=20
seed=0
agent_cfg_file="${ISAACLAB_DIR}/source/isaaclab_tasks/isaaclab_tasks/direct/shadow_hand/agents/rl_games_ppo_cfg.yaml"

read -r horizon_length minibatch_size < <(
    python - "${agent_cfg_file}" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as stream:
    cfg = yaml.safe_load(stream)["params"]["config"]
print(int(cfg["horizon_length"]), int(cfg["minibatch_size"]))
PY
)
rollout_batch_size=$((num_envs * horizon_length))
total_frames=$((rollout_batch_size * max_iterations))

if (( rollout_batch_size % minibatch_size != 0 )); then
    echo "incompatible_batch: rollout_batch_size=${rollout_batch_size}, minibatch_size=${minibatch_size}" >&2
    exit 2
fi

echo "record_type=shadow_expert_benchmark"
echo "timestamp=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname -f)"
echo "partition=${SLURM_JOB_PARTITION}"
echo "isaaclab_commit=$(git rev-parse HEAD)"
echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd ';')"
echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | paste -sd ';')"
echo "torch_cuda_available=$(python -c 'import torch; print(torch.cuda.is_available())')"
echo "torch_cuda_build=$(python -c 'import torch; print(torch.version.cuda)')"
echo "num_envs=${num_envs}"
echo "horizon_length=${horizon_length}"
echo "rollout_batch_size=${rollout_batch_size}"
echo "minibatch_size=${minibatch_size}"
echo "max_iterations=${max_iterations}"
echo "total_frames=${total_frames}"
echo "seed=${seed}"

start_epoch=$(date +%s)
run_log="${ISAACLAB_DIR}/logs/expert-benchmark/shadow-expert-benchmark-${SLURM_JOB_ID}.log"
gpu_log="${ISAACLAB_DIR}/logs/expert-benchmark/shadow-expert-benchmark-${SLURM_JOB_ID}-gpu.csv"

nvidia-smi \
    --query-gpu=timestamp,name,memory.used,utilization.gpu \
    --format=csv,noheader,nounits \
    --loop=2 > "${gpu_log}" &
monitor_pid=$!

set +e
/usr/bin/time -p timeout --signal=INT --kill-after=30s 1500s \
    ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Repose-Cube-Shadow-Direct-v0 \
    --num_envs "${num_envs}" \
    --max_iterations "${max_iterations}" \
    --seed "${seed}" \
    --headless \
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
python - "${run_log}" <<'PY'
import re
import statistics
import sys

text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
fps = [int(value) for value in re.findall(r"fps total:\s*(\d+)", text)]
steady = fps[2:] if len(fps) > 2 else fps
print(f"reported_iteration_count={len(fps)}")
if steady:
    print(f"steady_total_fps_median={statistics.median(steady):.1f}")
    print(f"steady_total_fps_mean={statistics.mean(steady):.1f}")
checkpoints = re.findall(r"saving checkpoint '([^']+)'", text)
print(f"last_checkpoint={checkpoints[-1] if checkpoints else 'not_reported'}")
PY

if [[ "${command_status}" -eq 0 ]]; then
    echo "expert_benchmark_status=PASS"
else
    echo "expert_benchmark_status=FAIL"
    exit "${command_status}"
fi
