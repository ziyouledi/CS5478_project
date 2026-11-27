#!/usr/bin/env bash
#SBATCH --job-name=shadow-video
#SBATCH --partition=gpu,gpu-long
#SBATCH --gpus=1
#SBATCH --constraint=xgpf
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Demo videos, one run per array task. VIDEO_RUN_DIRS is a space-separated list of experiment
# directories (each with nn/ and, for students, params/intent.json); task i films entry i.
# VIDEO_TASK selects the gym task (default Stage 1; Isaac-Repose-Cube-Shadow-ZAxis-Direct-v0 for M2).
# Optional: VIDEO_ITERATION (periodic checkpoint, default last), VIDEO_INTENT (runs without
# intent.json, e.g. blind for the expert), VIDEO_STEPS (default 600 = one episode),
# VIDEO_EYE / VIDEO_LOOKAT ("x y z" in the env-0 frame), VIDEO_NUM_ENVS (default 4; 1 keeps
# neighbouring hands out of the shot), VIDEO_ENV_INDEX (env the camera films, default 0; "median" =
# record a first pass, pick the env whose clip successes are closest to the median over all envs with
# select_median_env.py, then film that env in a second pass and check the two traces are identical).
# VIDEO_INTENT_DELAY=d plays the policy with an intent delay of d control steps instead of its trained
# delay (evaluation-only delay, as EVAL_INTENT_DELAY in 17); clip name and output dir get -delayD.
# VIDEO_CHECKPOINT=<pth> films that file instead of a periodic checkpoint of the run directory.
# VIDEO_TURN_CHECKPOINT=<pth> (VIDEO_TASK=Isaac-Switch-HingedCube-Shadow-Direct-v0, SWITCH_CYCLE): two-expert
# handoff, the face-turn expert in turn phases and VIDEO_CHECKPOINT (Stage 1 expert) otherwise.
# VIDEO_NAME replaces the run directory's basename in the clip name and output dir (for RL-Games log
# dirs named by date). Hinged-cube / face-turn tasks need HINGE_REFERENCE_JSON and read
# HINGE_EFFORT_LIMIT and FACE_TURN_HOLD_STEPS (see hinged_cube_env.py); the trace then also holds
# hinge_angle and hinge_goal.

set -Eeuo pipefail

: "${ISAACLAB_DIR:?Submit with ISAACLAB_DIR and VENV_DIR}"
: "${VENV_DIR:?Submit with ISAACLAB_DIR and VENV_DIR}"
: "${VIDEO_RUN_DIRS:?Submit with VIDEO_RUN_DIRS (space-separated experiment directories)}"

source "${VENV_DIR}/bin/activate"
cd "${ISAACLAB_DIR}"

export PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA=YES
export XDG_CACHE_HOME="${HOME}/data/low-dim-intent/cache/xdg"

read -r -a run_dirs <<< "${VIDEO_RUN_DIRS}"
task_id="${SLURM_ARRAY_TASK_ID:-0}"
if (( task_id >= ${#run_dirs[@]} )); then
    echo "array index ${task_id} is outside ${#run_dirs[@]} run directories" >&2
    exit 2
fi
run_dir="${run_dirs[${task_id}]}"
task="${VIDEO_TASK:-Isaac-Repose-Cube-Shadow-Direct-v0}"
job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${task_id}"
delay_tag="${VIDEO_INTENT_DELAY:+-delay${VIDEO_INTENT_DELAY}}"
base_name="${VIDEO_NAME:-$(basename "${run_dir}")}"
output_dir="${HOME}/data/low-dim-intent/videos/${base_name}${delay_tag}-job${job_tag}"
recorder="${SLURM_SUBMIT_DIR}/record_videos.py"

test -d "${run_dir}/nn"
test -f "${recorder}"
mkdir -p "${output_dir}" "${XDG_CACHE_HOME}"
cp "${SLURM_SUBMIT_DIR}"/{record_videos.py,select_median_env.py,student_eval_utils.py,intent_env_wrapper.py,intent_modules.py,m2_tasks.py,shadow_zaxis_env.py,hinged_cube_tasks.py,hinged_cube_env.py,21_record_videos.sbatch} "${output_dir}/"

echo "record_type=shadow_video_job"
echo "timestamp=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID}"
echo "array_task_id=${task_id}"
echo "node=$(hostname -f)"
echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd ';')"
echo "run_dir=${run_dir}"
echo "task=${task}"
echo "output_dir=${output_dir}"
if [[ "${task}" == *HingedCube* ]]; then
    : "${HINGE_REFERENCE_JSON:?Hinged-cube tasks need HINGE_REFERENCE_JSON}"
    export HINGE_ASSET_DIR="${output_dir}/asset"
    echo "hinge_reference_json=${HINGE_REFERENCE_JSON}"
    echo "hinge_effort_limit=${HINGE_EFFORT_LIMIT:-0.1}"
    echo "face_turn_hold_steps=${FACE_TURN_HOLD_STEPS:-1}"
fi

video_args=(
    --task "${task}"
    --run_dir "${run_dir}"
    --video_steps "${VIDEO_STEPS:-600}"
    --headless
)
env_index="${VIDEO_ENV_INDEX:-0}"
echo "env_index_request=${env_index}"
if [[ -n "${VIDEO_NUM_ENVS:-}" ]]; then video_args+=(--num_envs "${VIDEO_NUM_ENVS}"); fi
if [[ -n "${VIDEO_CHECKPOINT:-}" ]]; then test -f "${VIDEO_CHECKPOINT}"; video_args+=(--checkpoint "${VIDEO_CHECKPOINT}"); fi
if [[ -n "${VIDEO_TURN_CHECKPOINT:-}" ]]; then test -f "${VIDEO_TURN_CHECKPOINT}"; video_args+=(--turn_checkpoint "${VIDEO_TURN_CHECKPOINT}"); fi
if [[ -n "${VIDEO_ITERATION:-}" ]]; then video_args+=(--iteration "${VIDEO_ITERATION}"); fi
if [[ -n "${VIDEO_INTENT:-}" ]]; then video_args+=(--intent "${VIDEO_INTENT}"); fi
if [[ -n "${VIDEO_EYE:-}" ]]; then read -r -a eye <<< "${VIDEO_EYE}"; video_args+=(--eye "${eye[@]}"); fi
if [[ -n "${VIDEO_LOOKAT:-}" ]]; then read -r -a lookat <<< "${VIDEO_LOOKAT}"; video_args+=(--lookat "${lookat[@]}"); fi
if [[ -n "${VIDEO_INTENT_DELAY:-}" ]]; then video_args+=(--intent_delay "${VIDEO_INTENT_DELAY}"); fi
if [[ -n "${VIDEO_INTENT_DELAY:-}" || -n "${VIDEO_NAME:-}" ]]; then video_args+=(--name "${base_name}${delay_tag}"); fi
echo "intent_delay_override=${VIDEO_INTENT_DELAY:-none}"

record() {  # record <output_dir> <env_index>
    local out="$1" index="$2" status
    mkdir -p "${out}"
    set +e
    timeout --signal=INT --kill-after=30s 1500s \
        ./isaaclab.sh -p "${recorder}" "${video_args[@]}" --output_dir "${out}" --env_index "${index}" \
        2>&1 | tee "${out}/record.log"
    status=${PIPESTATUS[0]}
    set -e
    echo "command_status=${status} output=${out}"
    [[ "${status}" -eq 0 ]] && grep -q "video_record_status=PASS" "${out}/record.log"
}

start_epoch=$(date +%s)
status=PASS
name="${base_name}${delay_tag}"
# VIDEO_ENV_INDEX=median picks the env with select_median_env.py; VIDEO_ENV_INDEX=unmoved-turn (unified
# task) picks it with select_unmoved_turn_env.py. Both run a first pass, select, and film a second pass.
selector=""
case "${env_index}" in
    median) selector="select_median_env.py" ;;
    unmoved-turn) selector="select_unmoved_turn_env.py"; cp "${SLURM_SUBMIT_DIR}/${selector}" "${output_dir}/" ;;
esac
if [[ -n "${selector}" ]]; then
    if record "${output_dir}/select" 0; then
        selection="$(python "${output_dir}/${selector}" "${output_dir}/select/${name}_trace.npz")" || true
        echo "median_${selection}"
        env_index="$(sed -E 's/^selected_env=([0-9]+).*/\1/' <<< "${selection}")"
        if ! [[ "${env_index}" =~ ^[0-9]+$ ]]; then
            echo "env selection failed: ${selection}"; status=FAIL; env_index=0
        elif record "${output_dir}" "${env_index}"; then
            # Same seed and envs, so the run must repeat; only the camera differs. The criterion is the
            # success counters (what the selection used); max cube-orientation difference is printed too.
            if python -c 'import sys, numpy as np; a, b = (np.load(p) for p in sys.argv[1:]); d = float(np.abs(a["object_rot"] - b["object_rot"]).max()); print(f"median_pass_max_object_rot_diff={d:.3g}"); sys.exit(0 if np.array_equal(a["successes"], b["successes"]) else 1)' \
                "${output_dir}/select/${name}_trace.npz" "${output_dir}/${name}_trace.npz"; then
                echo "median_pass_determinism=PASS"
            else
                echo "median_pass_determinism=FAIL"; status=FAIL
            fi
        else
            status=FAIL
        fi
    else
        status=FAIL
    fi
else
    record "${output_dir}" "${env_index}" || status=FAIL
fi
echo "env_index=${env_index}"
echo "elapsed_seconds=$(( $(date +%s) - start_epoch ))"
echo "video_job_status=${status}"
[[ "${status}" == PASS ]]
