#!/usr/bin/env bash
#SBATCH --job-name=shadow-expert
#SBATCH --partition=gpu-long
#SBATCH --gpus=1
#SBATCH --constraint=xgpf
#SBATCH --time=06:00:00
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

num_envs=2048
seed=0
# EXPERT_TASK=m2 trains the M2 task (shadow_zaxis_env.py); unset keeps the Stage 1 command unchanged.
case "${EXPERT_TASK:-stage1}" in
    stage1) task=Isaac-Repose-Cube-Shadow-Direct-v0; run_prefix=expert; launcher=() ;;
    m2) task=Isaac-Repose-Cube-Shadow-ZAxis-Direct-v0; run_prefix=expert-m2
        launcher=("${SLURM_SUBMIT_DIR}/run_with_m2_tasks.py") ;;
    *) echo "unknown EXPERT_TASK=${EXPERT_TASK}" >&2; exit 2 ;;
esac
agent_cfg_file="${ISAACLAB_DIR}/source/isaaclab_tasks/isaaclab_tasks/direct/shadow_hand/agents/rl_games_ppo_cfg.yaml"

read -r horizon_length minibatch_size configured_max_epochs save_frequency < <(
    python - "${agent_cfg_file}" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as stream:
    cfg = yaml.safe_load(stream)["params"]["config"]
print(
    int(cfg["horizon_length"]),
    int(cfg["minibatch_size"]),
    int(cfg["max_epochs"]),
    int(cfg.get("save_frequency", -1)),
)
PY
)
max_iterations="${EXPERT_ITERATIONS:-${configured_max_epochs}}"
rollout_batch_size=$((num_envs * horizon_length))
total_frames=$((rollout_batch_size * max_iterations))

if (( rollout_batch_size % minibatch_size != 0 )); then
    echo "incompatible_batch: rollout_batch_size=${rollout_batch_size}, minibatch_size=${minibatch_size}" >&2
    exit 2
fi

run_root="${HOME}/data/low-dim-intent/runs/${run_prefix}-seed${seed}-job${SLURM_JOB_ID}"
metadata_dir="${HOME}/data/low-dim-intent/metadata/${run_prefix}-seed${seed}-job${SLURM_JOB_ID}"
mkdir -p "${run_root}" "${metadata_dir}" "${XDG_CACHE_HOME}"
cp "${agent_cfg_file}" "${metadata_dir}/rl_games_ppo_cfg.yaml"
cp "${SLURM_SUBMIT_DIR}/08_train_expert.sbatch" "${metadata_dir}/08_train_expert.sbatch"
if [[ "${EXPERT_TASK:-stage1}" == m2 ]]; then
    cp "${SLURM_SUBMIT_DIR}"/{m2_tasks.py,shadow_zaxis_env.py,run_with_m2_tasks.py} "${metadata_dir}/"
fi
scontrol show job "${SLURM_JOB_ID}" > "${metadata_dir}/slurm-job.txt"

echo "record_type=shadow_expert_train"
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
echo "configured_max_epochs=${configured_max_epochs}"
echo "max_iterations=${max_iterations}"
echo "total_frames=${total_frames}"
echo "save_frequency=${save_frequency}"
echo "seed=${seed}"
echo "task=${task}"
echo "resume_checkpoint=${CHECKPOINT:-none}"
echo "run_root=${run_root}"
echo "metadata_dir=${metadata_dir}"

run_log="${run_root}/train.log"
gpu_log="${run_root}/gpu.csv"
start_epoch=$(date +%s)

nvidia-smi \
    --query-gpu=timestamp,name,memory.used,utilization.gpu \
    --format=csv,noheader,nounits \
    --loop=5 > "${gpu_log}" &
monitor_pid=$!

train_args=(
    --task "${task}"
    --num_envs "${num_envs}"
    --max_iterations "${max_iterations}"
    --seed "${seed}"
    --headless
)
if [[ -n "${CHECKPOINT:-}" ]]; then
    train_args+=(--checkpoint "${CHECKPOINT}")
fi
if [[ "${EXPERT_TASK:-stage1}" == m2 ]]; then
    train_args+=(agent.params.config.name=shadow_hand_zaxis)
fi

set +e
/usr/bin/time -p timeout --signal=INT --kill-after=60s 20700s \
    ./isaaclab.sh -p "${launcher[@]}" scripts/reinforcement_learning/rl_games/train.py "${train_args[@]}" \
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
python - "${run_log}" "${ISAACLAB_DIR}" <<'PY'
import os
import re
import statistics
import sys

text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
fps = [int(value) for value in re.findall(r"fps total:\s*(\d+)", text)]
steady = fps[5:] if len(fps) > 5 else fps
print(f"reported_iteration_count={len(fps)}")
if steady:
    print(f"steady_total_fps_median={statistics.median(steady):.1f}")
    print(f"steady_total_fps_mean={statistics.mean(steady):.1f}")
checkpoints = re.findall(r"saving checkpoint '([^']+)'", text)
checkpoint = checkpoints[-1] if checkpoints else "not_reported"
if checkpoint != "not_reported" and not os.path.isabs(checkpoint):
    checkpoint = os.path.join(sys.argv[2], checkpoint)
print(f"last_checkpoint={checkpoint}")
PY

cp "${run_log}" "${metadata_dir}/train.log"
cp "${gpu_log}" "${metadata_dir}/gpu.csv"

if [[ "${command_status}" -eq 0 ]]; then
    echo "expert_train_status=PASS"
else
    echo "expert_train_status=FAIL"
    exit "${command_status}"
fi
