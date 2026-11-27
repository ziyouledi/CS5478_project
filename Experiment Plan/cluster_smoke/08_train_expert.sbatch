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

num_envs="${EXPERT_NUM_ENVS:-2048}"
seed="${EXPERT_SEED:-0}"
# EXPERT_TASK=m2 trains the M2 task (shadow_zaxis_env.py); unset keeps the Stage 1 command unchanged.
case "${EXPERT_TASK:-stage1}" in
    stage1) task=Isaac-Repose-Cube-Shadow-Direct-v0; run_prefix=expert; launcher=() ;;
    m2) task=Isaac-Repose-Cube-Shadow-ZAxis-Direct-v0; run_prefix=expert-m2
        launcher=("${SLURM_SUBMIT_DIR}/run_with_m2_tasks.py") ;;
    faceturn) task=Isaac-FaceTurn-HingedCube-Shadow-Direct-v0
        # Face-turn task on the hinged cube (hinged_cube_env.py). Needs HINGE_REFERENCE_JSON; the
        # detent torque cap comes from HINGE_EFFORT_LIMIT (N m).
        : "${HINGE_REFERENCE_JSON:?EXPERT_TASK=faceturn needs HINGE_REFERENCE_JSON}"
        export HINGE_EFFORT_LIMIT="${HINGE_EFFORT_LIMIT:-0.1}"
        export FACE_TURN_HOLD_STEPS="${FACE_TURN_HOLD_STEPS:-1}"
        faceturn_tag="cap${HINGE_EFFORT_LIMIT}"
        if [[ "${FACE_TURN_HOLD_STEPS}" != 1 ]]; then faceturn_tag+="_hold${FACE_TURN_HOLD_STEPS}"; fi
        # FACE_TURN_START_STATES: npz of start states left by another skill (see hinged_cube_env.py)
        if [[ -n "${FACE_TURN_START_STATES:-}" ]]; then faceturn_tag+="_handoff"; fi
        run_prefix="expert-faceturn-${faceturn_tag}"
        launcher=("${SLURM_SUBMIT_DIR}/run_with_m2_tasks.py") ;;
    unified) task=Isaac-Unified-HingedCube-Shadow-Direct-v0
        # One policy for reorientation and face turns (hinged_cube_env.py: ShadowHandUnifiedEnv).
        : "${HINGE_REFERENCE_JSON:?EXPERT_TASK=unified needs HINGE_REFERENCE_JSON}"
        export HINGE_EFFORT_LIMIT="${HINGE_EFFORT_LIMIT:-0.05}"
        export FACE_TURN_HOLD_STEPS="${FACE_TURN_HOLD_STEPS:-10}"
        faceturn_tag="cap${HINGE_EFFORT_LIMIT}_hold${FACE_TURN_HOLD_STEPS}"
        if [[ -n "${UNIFIED_TAG:-}" ]]; then faceturn_tag+="_${UNIFIED_TAG}"; fi
        run_prefix="expert-unified-${faceturn_tag}"
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
if [[ "${EXPERT_TASK:-stage1}" == faceturn || "${EXPERT_TASK:-stage1}" == unified ]]; then
    cp "${SLURM_SUBMIT_DIR}"/{hinged_cube_tasks.py,hinged_cube_env.py,m2_tasks.py,run_with_m2_tasks.py} "${metadata_dir}/"
    cp "${HINGE_REFERENCE_JSON}" "${metadata_dir}/hinge_reference.json"
    export HINGE_ASSET_DIR="${run_root}/asset"
    echo "hinge_reference_json=${HINGE_REFERENCE_JSON}"
    echo "hinge_effort_limit=${HINGE_EFFORT_LIMIT}"
    echo "face_turn_hold_steps=${FACE_TURN_HOLD_STEPS}"
    echo "face_turn_start_states=${FACE_TURN_START_STATES:-none}"
    echo "face_turn_start_fraction=${FACE_TURN_START_FRACTION:-0.5}"
    if [[ -n "${FACE_TURN_START_STATES:-}" ]]; then
        echo "face_turn_start_states_sha256=$(sha256sum "${FACE_TURN_START_STATES}" | cut -d' ' -f1)"
    fi
    echo "hinge_stiffness=${HINGE_STIFFNESS:-5.0}"
    echo "hinge_damping=${HINGE_DAMPING:-0.02}"
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
if [[ "${EXPERT_TASK:-stage1}" == faceturn ]]; then
    train_args+=("agent.params.config.name=shadow_hand_faceturn_${faceturn_tag}")
fi
if [[ "${EXPERT_TASK:-stage1}" == unified ]]; then
    train_args+=("agent.params.config.name=shadow_hand_unified_${faceturn_tag}")
    echo "unified_tag=${UNIFIED_TAG:-none}"
    echo "unified_turn_orient_tol=${UNIFIED_TURN_ORIENT_TOL:-0.4}"
    echo "unified_max_phase_steps=${UNIFIED_MAX_PHASE_STEPS:-120}"
    echo "unified_timeout_terminates=${UNIFIED_TIMEOUT_TERMINATES:-1}"
    echo "unified_hinge_prog_scale=${UNIFIED_HINGE_PROG_SCALE:-30.0}"
    echo "unified_orient_prog_scale=${UNIFIED_ORIENT_PROG_SCALE:-15.0}"
    echo "unified_hold_prog_scale=${UNIFIED_HOLD_PROG_SCALE:-5.0}"
    echo "unified_alive_bonus=${UNIFIED_ALIVE_BONUS:-0.5}"
    echo "unified_fall_penalty=${UNIFIED_FALL_PENALTY:--15.0}"
    echo "unified_extra_obs=${UNIFIED_EXTRA_OBS:-1}"
    echo "unified_start_states=${UNIFIED_START_STATES:-none}"
    echo "unified_start_fraction=${UNIFIED_START_FRACTION:-0.5}"
    if [[ -n "${UNIFIED_START_STATES:-}" ]]; then
        echo "unified_start_states_sha256=$(sha256sum "${UNIFIED_START_STATES}" | cut -d' ' -f1)"
    fi
fi

set +e
/usr/bin/time -p timeout --signal=INT --kill-after=60s "${EXPERT_TIMEOUT_S:-20700}s" \
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
