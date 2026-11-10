#!/usr/bin/env bash
#SBATCH --job-name=shadow-video
#SBATCH --partition=gpu,gpu-long
#SBATCH --gpus=1
#SBATCH --constraint=xgpf
#SBATCH --time=00:30:00
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
# neighbouring hands out of the shot).

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
output_dir="${HOME}/data/low-dim-intent/videos/$(basename "${run_dir}")-job${job_tag}"
recorder="${SLURM_SUBMIT_DIR}/record_videos.py"

test -d "${run_dir}/nn"
test -f "${recorder}"
mkdir -p "${output_dir}" "${XDG_CACHE_HOME}"
cp "${SLURM_SUBMIT_DIR}"/{record_videos.py,student_eval_utils.py,intent_env_wrapper.py,intent_modules.py,m2_tasks.py,shadow_zaxis_env.py,21_record_videos.sbatch} "${output_dir}/"

echo "record_type=shadow_video_job"
echo "timestamp=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID}"
echo "array_task_id=${task_id}"
echo "node=$(hostname -f)"
echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd ';')"
echo "run_dir=${run_dir}"
echo "task=${task}"
echo "output_dir=${output_dir}"

video_args=(
    --task "${task}"
    --run_dir "${run_dir}"
    --video_steps "${VIDEO_STEPS:-600}"
    --output_dir "${output_dir}"
    --headless
)
if [[ -n "${VIDEO_NUM_ENVS:-}" ]]; then video_args+=(--num_envs "${VIDEO_NUM_ENVS}"); fi
if [[ -n "${VIDEO_ITERATION:-}" ]]; then video_args+=(--iteration "${VIDEO_ITERATION}"); fi
if [[ -n "${VIDEO_INTENT:-}" ]]; then video_args+=(--intent "${VIDEO_INTENT}"); fi
if [[ -n "${VIDEO_EYE:-}" ]]; then read -r -a eye <<< "${VIDEO_EYE}"; video_args+=(--eye "${eye[@]}"); fi
if [[ -n "${VIDEO_LOOKAT:-}" ]]; then read -r -a lookat <<< "${VIDEO_LOOKAT}"; video_args+=(--lookat "${lookat[@]}"); fi

start_epoch=$(date +%s)
set +e
timeout --signal=INT --kill-after=30s 1500s \
    ./isaaclab.sh -p "${recorder}" "${video_args[@]}" 2>&1 | tee "${output_dir}/record.log"
command_status=${PIPESTATUS[0]}
set -e
echo "elapsed_seconds=$(( $(date +%s) - start_epoch ))"
echo "command_status=${command_status}"

if [[ "${command_status}" -eq 0 ]] && grep -q "video_record_status=PASS" "${output_dir}/record.log"; then
    echo "video_job_status=PASS"
else
    echo "video_job_status=FAIL"
    exit "${command_status:-1}"
fi
