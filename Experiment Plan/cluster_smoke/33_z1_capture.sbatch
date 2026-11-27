#!/usr/bin/env bash
#SBATCH --job-name=z1-capture
#SBATCH --partition=gpu,gpu-long
#SBATCH --gpus=1
#SBATCH --constraint=xgpf
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Z1 frames (GOAL_LEVEL_RL_PLAN.md "Z1 的渲染设定"): frozen unified expert v4 on the Z1 rendering task
# (z1_render_env.py: coloured face plates, fixed per-env TiledCameras), frames rendered with and without hands.
# Defaults are the dry run: 4 envs, 130 steps, captures at steps 60 and 120, 5 candidate
# cameras, stability check. Override: Z1_NUM_ENVS, Z1_STEPS, Z1_CAPTURE_START, Z1_CAPTURE_EVERY, Z1_CAMERAS,
# Z1_IMAGE, Z1_FOCAL, Z1_TARGET, Z1_ENV_SPACING, Z1_SEED, Z1_EXTRA_ARGS (e.g. --no_png), Z1_TAG.
# Physics env = v4 evaluation env of 31_switch_eval.sbatch (cap 0.05, hold 10, random goal order, no start states).

set -Eeuo pipefail
: "${ISAACLAB_DIR:?Submit with ISAACLAB_DIR and VENV_DIR}"
: "${VENV_DIR:?Submit with ISAACLAB_DIR and VENV_DIR}"

source "${VENV_DIR}/bin/activate"
cd "${ISAACLAB_DIR}"

export PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA=YES
export XDG_CACHE_HOME="${HOME}/data/low-dim-intent/cache/xdg"
export HINGE_REFERENCE_JSON="${HINGE_REFERENCE_JSON:-${HOME}/data/low-dim-intent/runs/hinged-cube-eval-job1141499/dexcube.json}"
export HINGE_EFFORT_LIMIT=0.05 FACE_TURN_HOLD_STEPS=10
unset UNIFIED_CYCLE UNIFIED_START_STATES FACE_TURN_START_STATES HINGE_STIFFNESS HINGE_DAMPING
export Z1_CAMERAS="${Z1_CAMERAS:-c1:-50:38:0.33,s040:40:30:0.33,s130:130:30:0.33,s220:-140:30:0.33,top:-50:85:0.33}"
export Z1_CAMERAS="${Z1_CAMERAS//;/,}"
export Z1_IMAGE="${Z1_IMAGE:-512}" Z1_FOCAL="${Z1_FOCAL:-24}" Z1_TARGET="${Z1_TARGET:-0 -0.39 0.6}"

checkpoint="${Z1_CHECKPOINT:-${HOME}/data/low-dim-intent/checkpoints/expert-unified-cap0.05_hold10_v4-seed0/frozen/expert-unified-v4-seed0.pth}"
expected_sha256=ca3d9367
test -f "${checkpoint}"
tag="${Z1_TAG:-dry}"
# Array jobs: task i uses seed Z1_SEED + i and its own output dir. A T4 runs out of render memory at
# 128 envs x 5 cameras x 512^2 (job 1163441, then hung until the timeout), so large captures are split.
task_id="${SLURM_ARRAY_TASK_ID:-0}"
seed=$(( ${Z1_SEED:-20261118} + task_id ))
job_tag="${SLURM_ARRAY_JOB_ID:+${SLURM_ARRAY_JOB_ID}_${task_id}}"
output_dir="${HOME}/data/low-dim-intent/runs/z1-${tag}-job${job_tag:-${SLURM_JOB_ID}}"
mkdir -p "${output_dir}/code" "${XDG_CACHE_HOME}"
export HINGE_ASSET_DIR="${output_dir}/asset"
for file in z1_capture.py z1_render_env.py z1_tasks.py hinged_cube_env.py hinged_cube_tasks.py intent_modules.py \
        33_z1_capture.sbatch; do
    cp "${SLURM_SUBMIT_DIR}/${file}" "${output_dir}/code/${file}"
done

echo "record_type=z1_capture_job"
echo "timestamp=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname -f)"
echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd ';')"
echo "checkpoint=${checkpoint}"
checkpoint_sha256="$(sha256sum "${checkpoint}" | cut -d' ' -f1)"
echo "checkpoint_sha256=${checkpoint_sha256}"
[[ "${checkpoint_sha256}" == ${expected_sha256}* ]] || [[ -n "${Z1_CHECKPOINT:-}" ]] || {
    echo "default checkpoint is not the frozen v4 (sha256 ${expected_sha256}...)" >&2; exit 2; }
echo "z1_cameras=${Z1_CAMERAS}"
echo "seed=${seed}"
echo "output_dir=${output_dir}"

read -r -a extra <<< "${Z1_EXTRA_ARGS:---stability_check}"
start_epoch=$(date +%s)
set +e
timeout --signal=INT --kill-after=30s "${Z1_TIMEOUT_S:-1500}" \
    ./isaaclab.sh -p "${SLURM_SUBMIT_DIR}/z1_capture.py" \
        --checkpoint "${checkpoint}" \
        --num_envs "${Z1_NUM_ENVS:-4}" \
        --steps "${Z1_STEPS:-130}" \
        --capture_start "${Z1_CAPTURE_START:-60}" \
        --capture_every "${Z1_CAPTURE_EVERY:-60}" \
        --env_spacing "${Z1_ENV_SPACING:-5.0}" \
        --seed "${seed}" \
        --output_dir "${output_dir}" \
        --headless "${extra[@]}" \
    2>&1 | tee "${output_dir}/capture.log" | while IFS= read -r line; do
        printf '%s\n' "${line}"
        # Out-of-memory in the renderer leaves Kit hanging (job 1163441): stop at the first such line.
        if [[ "${line}" == *ERROR_OUT_OF_DEVICE_MEMORY* ]]; then
            echo "z1_render_oom: ${line}"; pkill -INT -f "z1_capture.py --checkpoint" || true; break
        fi
    done
status=${PIPESTATUS[0]}
set -e
echo "command_status=${status}"
echo "elapsed_seconds=$(( $(date +%s) - start_epoch ))"
grep -E "^z1_capture_status=" "${output_dir}/capture.log" || echo "z1_capture_status=MISSING"
[[ "${status}" -eq 0 ]] && grep -q "z1_capture_status=PASS" "${output_dir}/capture.log"
