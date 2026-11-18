#!/usr/bin/env bash
#SBATCH --job-name=shadow-student
#SBATCH --partition=gpu,gpu-long
#SBATCH --gpus=1
#SBATCH --constraint=xgpf
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# One student run per array task. Task index i maps to
#   intent = STUDENT_INTENTS[i / num_seeds], seed = STUDENT_SEEDS[i % num_seeds].
# Submit the stage-one grid (7 intents x 3 seeds) with --array=0-20%<max concurrent>.
# STUDENT_INTENT_HOLD=N (rate sweep) refreshes the intent every N episode steps; N != 1 adds -holdN to the run name.
# A STUDENT_INTENTS entry may carry its own hold as intent:N (e.g. "goal-4:1 hand-8:6"), overriding STUDENT_INTENT_HOLD.
# An entry intent:dD (e.g. "goal-4:d30") trains under an intent delay of D control steps instead (hold 1);
# the run name gets -delayD (rate sweep phase 3).
# STUDENT_ACTION_HOLD=N (rate sweep phase 4) applies a new policy action only every N steps; run name gets -aholdN.
# STUDENT_SEEDS overrides the seed list (default "0 1 2"), e.g. "3 4".
# STUDENT_RESUME_FROM (rate sweep phase 2) is a space-separated list of checkpoints, entry i for array task i;
# training resumes from it up to STUDENT_ITERATIONS (absolute epoch count) and the run name gets -resume.
# train_student.py checks that the checkpoint's run has the same intent, hold, seed, task and flags.
# STUDENT_PRINT_PLAN=1 prints the task -> (stage, intent, hold, seed, resume, name) mapping and exits.

set -Eeuo pipefail

: "${ISAACLAB_DIR:?Submit with ISAACLAB_DIR and VENV_DIR}"
: "${VENV_DIR:?Submit with ISAACLAB_DIR and VENV_DIR}"
: "${INTENT_PCA_DIR:?Submit with INTENT_PCA_DIR (a completed action-pca output directory)}"
: "${STUDENT_ITERATIONS:?Submit with STUDENT_ITERATIONS (PPO iterations per run)}"

source "${VENV_DIR}/bin/activate"
cd "${ISAACLAB_DIR}"

export PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA=YES
export XDG_CACHE_HOME="${HOME}/data/low-dim-intent/cache/xdg"

read -r -a intents <<< "${STUDENT_INTENTS:-blind wrist-2 hand-2 hand-4 hand-20 obj-1 obj-4}"
read -r -a seeds <<< "${STUDENT_SEEDS:-0 1 2}"
stage="${STUDENT_STAGE:-stage1}"
obj_lookahead="${OBJ_LOOKAHEAD:-0.5}"
intent_hold="${STUDENT_INTENT_HOLD:-1}"
num_envs=2048
task_id="${SLURM_ARRAY_TASK_ID:-0}"
if (( task_id >= ${#intents[@]} * ${#seeds[@]} )); then
    echo "array index ${task_id} is outside ${#intents[@]} intents x ${#seeds[@]} seeds" >&2
    exit 2
fi
intent="${intents[$((task_id / ${#seeds[@]}))]}"
intent_delay=0
if [[ "${intent}" == *:d* ]]; then
    intent_delay="${intent##*:d}"
    intent="${intent%%:*}"
    intent_hold=1
elif [[ "${intent}" == *:* ]]; then
    intent_hold="${intent##*:}"
    intent="${intent%%:*}"
fi
if ! [[ "${intent_delay}" =~ ^[0-9]+$ ]]; then
    echo "invalid intent delay '${intent_delay}'" >&2
    exit 2
fi
if ! [[ "${intent_hold}" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid intent hold '${intent_hold}'" >&2
    exit 2
fi
seed="${seeds[$((task_id % ${#seeds[@]}))]}"
resume_from=""
if [[ -n "${STUDENT_RESUME_FROM:-}" ]]; then
    read -r -a resume_list <<< "${STUDENT_RESUME_FROM}"
    if (( ${#resume_list[@]} != ${#intents[@]} * ${#seeds[@]} )); then
        echo "STUDENT_RESUME_FROM has ${#resume_list[@]} entries, expected ${#intents[@]} x ${#seeds[@]}" >&2
        exit 2
    fi
    resume_from="${resume_list[${task_id}]}"
fi

# STUDENT_STAGE=m2 trains on the M2 task (shadow_zaxis_env.py) with the frozen M2 expert.
# STUDENT_STAGE=m1 is Stage 2 M1: Stage 1 task and expert, goal hidden from the actor, asymmetric
# critic. STUDENT_STAGE=m1-gate keeps the goal visible with the same asymmetric critic (M1 gate run).
# STUDENT_STAGE=m2-hidden is M1's setting on the M2 task and expert (rate sweep phase 2).
# STUDENT_STAGE=faceturn-hidden is M1's setting on the face-turn task (FACE_TURN_PLAN.md, E1).
student_flags=()
expert_file=""
case "${stage}" in
    m2) task=Isaac-Repose-Cube-Shadow-ZAxis-Direct-v0; expert_name=expert-m2-seed0 ;;
    m2-hidden) task=Isaac-Repose-Cube-Shadow-ZAxis-Direct-v0; expert_name=expert-m2-seed0
        student_flags=(--goal_hidden --asymmetric_critic) ;;
    m1) task=Isaac-Repose-Cube-Shadow-Direct-v0; expert_name=expert-seed0
        student_flags=(--goal_hidden --asymmetric_critic) ;;
    m1-gate) task=Isaac-Repose-Cube-Shadow-Direct-v0; expert_name=expert-seed0
        student_flags=(--asymmetric_critic) ;;
    stage1) task=Isaac-Repose-Cube-Shadow-Direct-v0; expert_name=expert-seed0 ;;
    faceturn-hidden) task=Isaac-FaceTurn-HingedCube-Shadow-Direct-v0
        # M1's setting on the face-turn task: goal hidden, asymmetric critic, and the actor also gets
        # (sin, cos) of the current hinge angle (intent_env_wrapper.py). Expert = the handoff-trained one.
        expert_name=expert-faceturn-cap0.05_hold10_handoff-seed0; expert_file=expert-faceturn-handoff-seed0.pth
        student_flags=(--goal_hidden --asymmetric_critic --hinge_angle_obs)
        # Face-turn task (hinged_cube_env.py) as its expert was trained and evaluated: torque cap 0.05 N m,
        # a turn counts after 10 held steps, flat start only.
        export HINGE_REFERENCE_JSON="${HINGE_REFERENCE_JSON:-${HOME}/data/low-dim-intent/runs/hinged-cube-eval-job1141499/dexcube.json}"
        export HINGE_EFFORT_LIMIT=0.05 FACE_TURN_HOLD_STEPS=10
        unset FACE_TURN_START_STATES ;;
    unified-hidden) task=Isaac-Unified-HingedCube-Shadow-Direct-v0
        # E3 (FACE_TURN_PLAN.md): students of the frozen v4 unified teacher on its training env
        # (unified_env.sh). Goal-hidden actor = unified_actor_obs (intent_env_wrapper.py), asymmetric critic.
        source "${SLURM_SUBMIT_DIR:-.}/unified_env.sh"
        expert_name="${unified_teacher_name}"; expert_file="${unified_teacher_file}"
        student_flags=(--goal_hidden --asymmetric_critic --hinge_angle_obs) ;;
    *) echo "unknown STUDENT_STAGE=${stage}" >&2; exit 2 ;;
esac
# The action PCA must come from the same task's expert rollout (M2 basis directories contain "expert-m2").
case "${stage}" in
    m2*) [[ "${INTENT_PCA_DIR}" == *expert-m2* ]] || { echo "STUDENT_STAGE=${stage} needs the M2 action PCA" >&2; exit 2; } ;;
    faceturn*) [[ "${INTENT_PCA_DIR}" == *expert-faceturn* ]] || { echo "STUDENT_STAGE=${stage} needs the face-turn action PCA" >&2; exit 2; } ;;
    unified*) [[ "${INTENT_PCA_DIR}" == *expert-unified* ]] || { echo "STUDENT_STAGE=${stage} needs the unified v4 action PCA" >&2; exit 2; } ;;
    *) [[ "${INTENT_PCA_DIR}" != *expert-m2* && "${INTENT_PCA_DIR}" != *expert-faceturn* && "${INTENT_PCA_DIR}" != *expert-unified* ]] || { echo "STUDENT_STAGE=${stage} must not use the M2, face-turn or unified action PCA" >&2; exit 2; } ;;
esac
job_tag="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-plan}}_${task_id}"
hold_tag=""
if [[ "${intent_hold}" != "1" ]]; then hold_tag="-hold${intent_hold}"; fi
if [[ "${intent_delay}" != "0" ]]; then hold_tag="-delay${intent_delay}"; fi
action_hold="${STUDENT_ACTION_HOLD:-1}"
if ! [[ "${action_hold}" =~ ^[1-9][0-9]*$ ]]; then echo "invalid action hold '${action_hold}'" >&2; exit 2; fi
if [[ "${action_hold}" != "1" ]]; then hold_tag="${hold_tag}-ahold${action_hold}"; fi
resume_tag=""
if [[ -n "${resume_from}" ]]; then resume_tag="-resume"; fi
name="student-${stage}-${intent}${hold_tag}-seed${seed}${resume_tag}"
if [[ "${STUDENT_PRINT_PLAN:-0}" == "1" ]]; then
    echo "plan task=${task_id} stage=${stage} task_name=${task} intent=${intent} hold=${intent_hold} delay=${intent_delay} action_hold=${action_hold} seed=${seed} iterations=${STUDENT_ITERATIONS} resume=${resume_from:-none} name=${name}"
    exit 0
fi
frozen_dir="${HOME}/data/low-dim-intent/checkpoints/${expert_name}/frozen"
expert_file="${expert_file:-${expert_name}.pth}"
checkpoint="${frozen_dir}/${expert_file}"
expert_sha256=$(awk -v name="${expert_file}" '$2 == name {print $1}' "${frozen_dir}/SHA256SUMS")
# joint-K intents use the joint-position basis (20_fit_joint_basis.sbatch); all others the action PCA.
case "${intent}" in
    joint-*) pca_basis="${JOINT_PCA_DIR:?joint-K intents need JOINT_PCA_DIR}/joint_basis.npz" ;;
    *) pca_basis="${INTENT_PCA_DIR}/pca_basis.npz" ;;
esac
run_root="${HOME}/data/low-dim-intent/runs/${name}-job${job_tag}"
metadata_dir="${HOME}/data/low-dim-intent/metadata/${name}-job${job_tag}"
trainer="${SLURM_SUBMIT_DIR}/train_student.py"

test -f "${checkpoint}"
test -n "${expert_sha256}"
test -f "${pca_basis}"
test -f "${trainer}"
resume_args=()
if [[ -n "${resume_from}" ]]; then
    test -f "${resume_from}"
    resume_args=(--checkpoint "${resume_from}")
fi
mkdir -p "${run_root}" "${metadata_dir}" "${XDG_CACHE_HOME}"
cp "${SLURM_SUBMIT_DIR}"/{train_student.py,intent_env_wrapper.py,intent_modules.py,student_eval_utils.py,m2_tasks.py,shadow_zaxis_env.py,16_train_student.sbatch} "${metadata_dir}/"
if [[ "${stage}" == faceturn* || "${stage}" == unified* ]]; then
    cp "${SLURM_SUBMIT_DIR}"/{hinged_cube_tasks.py,hinged_cube_env.py,unified_env.sh} "${metadata_dir}/"
    export HINGE_ASSET_DIR="${run_root}/asset"
    echo "hinge_reference_json=${HINGE_REFERENCE_JSON}"
    echo "hinge_effort_limit=${HINGE_EFFORT_LIMIT}"
    echo "face_turn_hold_steps=${FACE_TURN_HOLD_STEPS}"
    if [[ "${stage}" == unified* ]]; then
        env | grep -E '^UNIFIED_' | sort
        echo "unified_start_states_sha256=$(sha256sum "${UNIFIED_START_STATES}" | cut -d' ' -f1)"
    fi
fi
scontrol show job "${SLURM_JOB_ID}" > "${metadata_dir}/slurm-job.txt"

echo "record_type=shadow_student_train_job"
echo "timestamp=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID}"
echo "array_job_id=${SLURM_ARRAY_JOB_ID:-none}"
echo "array_task_id=${task_id}"
echo "node=$(hostname -f)"
echo "isaaclab_commit=$(git rev-parse HEAD)"
echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd ';')"
echo "stage=${stage}"
echo "task=${task}"
echo "student_flags=${student_flags[*]:-none}"
echo "expert_checkpoint=${checkpoint}"
echo "intent=${intent}"
echo "seed=${seed}"
echo "num_envs=${num_envs}"
echo "iterations=${STUDENT_ITERATIONS}"
echo "expert_sha256=${expert_sha256}"
echo "pca_basis=${pca_basis}"
echo "obj_lookahead=${obj_lookahead}"
echo "intent_hold=${intent_hold}"
echo "intent_delay=${intent_delay}"
echo "action_hold=${action_hold}"
echo "resume_from=${resume_from:-none}"
echo "run_root=${run_root}"

run_log="${run_root}/train.log"
gpu_log="${run_root}/gpu.csv"
nvidia-smi \
    --query-gpu=timestamp,name,memory.used,utilization.gpu \
    --format=csv,noheader,nounits \
    --loop=30 > "${gpu_log}" &
monitor_pid=$!
trap 'kill "${monitor_pid}" 2>/dev/null || true; wait "${monitor_pid}" 2>/dev/null || true' EXIT

start_epoch=$(date +%s)
set +e
/usr/bin/time -p timeout --signal=INT --kill-after=60s 10500s \
    ./isaaclab.sh -p "${trainer}" \
    --task "${task}" \
    --intent "${intent}" \
    --expert_checkpoint "${checkpoint}" \
    --expert_sha256 "${expert_sha256}" \
    --pca_basis "${pca_basis}" \
    "${student_flags[@]}" \
    --obj_lookahead "${obj_lookahead}" \
    --intent_hold "${intent_hold}" \
    --intent_delay "${intent_delay}" \
    --action_hold "${action_hold}" \
    "${resume_args[@]}" \
    --num_envs "${num_envs}" \
    --seed "${seed}" \
    --max_iterations "${STUDENT_ITERATIONS}" \
    --train_dir "${run_root}" \
    --experiment_name "${name}" \
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
awk -F, '
    {gsub(/ /, "", $3); if (($3 + 0) > max) max = $3 + 0}
    END {printf "peak_gpu_memory_mib=%d\n", max}
' "${gpu_log}"
python - "${run_log}" <<'PY'
import re
import statistics
import sys

text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
fps = [int(value) for value in re.findall(r"fps total:\s*(\d+)", text)]
steady = fps[5:] if len(fps) > 5 else fps
print(f"reported_iteration_count={len(fps)}")
if steady:
    print(f"steady_total_fps_median={statistics.median(steady):.1f}")
checkpoints = re.findall(r"saving checkpoint '([^']+)'", text)
print(f"last_checkpoint={checkpoints[-1] if checkpoints else 'not_reported'}")
PY
cp "${run_log}" "${metadata_dir}/train.log"
cp "${gpu_log}" "${metadata_dir}/gpu.csv"

if [[ "${command_status}" -eq 0 ]] && grep -q "student_train_status=PASS" "${run_log}"; then
    echo "student_job_status=PASS"
else
    echo "student_job_status=FAIL"
    exit "${command_status:-1}"
fi
