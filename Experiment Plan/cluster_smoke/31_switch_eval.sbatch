#!/usr/bin/env bash
#SBATCH --job-name=switch-eval
#SBATCH --partition=gpu,gpu-long
#SBATCH --gpus=1
#SBATCH --constraint=xgpf
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Task-switching test of two frozen experts on the hinged cube (evaluate_switch.py, no training).
# SWITCH_CONDITIONS: space-separated cycles, each a comma-separated list of phase kinds
# (any / up / up2 / turn, see ShadowHandSwitchEnv). One-kind cycles are the single-task baselines.
# Fixed by default: cap 0.05 N m, hold 10 steps, 1024 envs, 600 control steps, seed 20261006.
# SWITCH_CAPTURE=<kinds> also saves the end states of those phase kinds to <run_root>/<label>-states.npz
# (start states for training another skill, FACE_TURN_START_STATES in hinged_cube_env.py).
# UNIFIED_CHECKPOINT=<pth> evaluates one unified-task policy instead of the two experts; conditions are
# then UNIFIED_CYCLE values (any / up / turn, not starting with turn) or "random" (training order).
# Selector channel (two experts, FACE_TURN_PLAN.md E2'): a condition may end in @h<N> (message held N steps)
# or @d<D> (message delayed D steps), e.g. "up,turn@h6 any,up,turn@d30"; plain cycles = N 1, d 0.
# Controls: @only-reorient / @only-turn drive every phase with that one expert.
# STUDENT_RUN_DIR=<run dir with nn/ and params/intent.json> evaluates one unified-task student (FACE_TURN_PLAN.md E3)
# with the conditions of UNIFIED_CHECKPOINT; STUDENT_RUN_DIRS="<dir0> <dir1> ..." picks entry SLURM_ARRAY_TASK_ID
# (grid evaluation as an array, aftercorr on the training array). Phase timeout off and stock resets, as for v4.
# STUDENT_EVAL_ITERATIONS="1000 2000 ..." (students only) runs every condition on the periodic checkpoint of each
# iteration (label suffix -it<N>) instead of the last checkpoint.

set -Eeuo pipefail

: "${ISAACLAB_DIR:?Submit with ISAACLAB_DIR and VENV_DIR}"
: "${VENV_DIR:?Submit with ISAACLAB_DIR and VENV_DIR}"
: "${HINGE_REFERENCE_JSON:?Submit with HINGE_REFERENCE_JSON}"

source "${VENV_DIR}/bin/activate"
cd "${ISAACLAB_DIR}"

export PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA=YES
export XDG_CACHE_HOME="${HOME}/data/low-dim-intent/cache/xdg"
export HINGE_EFFORT_LIMIT="${HINGE_EFFORT_LIMIT:-0.05}"
export FACE_TURN_HOLD_STEPS="${FACE_TURN_HOLD_STEPS:-10}"

checkpoints="${HOME}/data/low-dim-intent/checkpoints"
reorient_checkpoint="${REORIENT_CHECKPOINT:-${checkpoints}/expert-seed0/frozen/expert-seed0.pth}"
turn_checkpoint="${TURN_CHECKPOINT:-${checkpoints}/expert-faceturn-cap0.05_hold10-seed0/frozen/expert-faceturn-seed0.pth}"
conditions="${SWITCH_CONDITIONS:-turn any up up2 up,turn up2,turn any,up,turn any,up2,turn}"
num_envs="${EVAL_NUM_ENVS:-1024}"
steps="${EVAL_STEPS:-600}"
seed="${EVAL_SEED:-20261006}"
job_tag="${SLURM_JOB_ID}"
if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then job_tag="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; fi
run_root="${HOME}/data/low-dim-intent/runs/switch-eval-job${job_tag}"
metadata_dir="${HOME}/data/low-dim-intent/metadata/switch-eval-job${job_tag}"
mkdir -p "${run_root}" "${metadata_dir}" "${XDG_CACHE_HOME}"
for file in evaluate_switch.py hinged_cube_env.py hinged_cube_tasks.py intent_modules.py intent_env_wrapper.py \
        student_eval_utils.py 31_switch_eval.sbatch; do
    cp "${SLURM_SUBMIT_DIR}/${file}" "${metadata_dir}/${file}"
done
scontrol show job "${SLURM_JOB_ID}" > "${metadata_dir}/slurm-job.txt"
test -f "${reorient_checkpoint}"
test -f "${turn_checkpoint}"
if [[ -z "${UNIFIED_CHECKPOINT:-}" && -n "${UNIFIED_TRAIN_LOG:-}" ]]; then
    UNIFIED_CHECKPOINT="$(grep '^last_checkpoint=' "${UNIFIED_TRAIN_LOG}" | tail -n 1 | cut -d= -f2-)"
fi
if [[ -z "${STUDENT_RUN_DIR:-}" && -n "${STUDENT_RUN_DIRS:-}" ]]; then
    read -r -a student_dirs <<< "${STUDENT_RUN_DIRS}"
    STUDENT_RUN_DIR="${student_dirs[${SLURM_ARRAY_TASK_ID:?STUDENT_RUN_DIRS needs an array job}]:?no entry for task ${SLURM_ARRAY_TASK_ID}}"
fi
if [[ -n "${STUDENT_RUN_DIR:-}" ]]; then
    [[ -z "${UNIFIED_CHECKPOINT:-}" ]] || { echo "STUDENT_RUN_DIR and UNIFIED_CHECKPOINT are exclusive" >&2; exit 2; }
    test -f "${STUDENT_RUN_DIR}/params/intent.json"
    unset UNIFIED_START_STATES UNIFIED_CYCLE
    export UNIFIED_MAX_PHASE_STEPS=0
    policy_args=(--student_run_dir "${STUDENT_RUN_DIR}")
    cp "${STUDENT_RUN_DIR}/params/intent.json" "${metadata_dir}/student-intent.json"
    echo "student_run_dir=${STUDENT_RUN_DIR}"
    echo "student_intent=$(python -c 'import json,sys; r=json.load(open(sys.argv[1])); print(r["intent"], "hold", r.get("intent_hold"), "seed", r.get("seed"))' "${STUDENT_RUN_DIR}/params/intent.json")"
elif [[ -n "${UNIFIED_CHECKPOINT:-}" ]]; then
    test -f "${UNIFIED_CHECKPOINT}"
    policy_args=(--unified_checkpoint "${UNIFIED_CHECKPOINT}")
    echo "unified_checkpoint=${UNIFIED_CHECKPOINT}"
    echo "unified_checkpoint_sha256=$(sha256sum "${UNIFIED_CHECKPOINT}" | cut -d' ' -f1)"
else
    policy_args=(--reorient_checkpoint "${reorient_checkpoint}" --turn_checkpoint "${turn_checkpoint}")
fi

echo "record_type=shadow_task_switch_eval"
echo "timestamp=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname -f)"
echo "isaaclab_commit=$(git rev-parse HEAD)"
echo "reorient_checkpoint=${reorient_checkpoint}"
echo "reorient_checkpoint_sha256=$(sha256sum "${reorient_checkpoint}" | cut -d' ' -f1)"
echo "turn_checkpoint=${turn_checkpoint}"
echo "turn_checkpoint_sha256=$(sha256sum "${turn_checkpoint}" | cut -d' ' -f1)"
echo "hinge_effort_limit=${HINGE_EFFORT_LIMIT}"
echo "face_turn_hold_steps=${FACE_TURN_HOLD_STEPS}"
echo "num_envs=${num_envs}"
echo "steps=${steps}"
echo "evaluation_seed=${seed}"
echo "conditions=${conditions}"
echo "run_root=${run_root}"

overall_status=0
items=()
for condition in ${conditions}; do
    if [[ -n "${STUDENT_EVAL_ITERATIONS:-}" ]]; then
        [[ -n "${STUDENT_RUN_DIR:-}" ]] || { echo "STUDENT_EVAL_ITERATIONS needs a student run" >&2; exit 2; }
        for it in ${STUDENT_EVAL_ITERATIONS}; do items+=("${condition}#${it}"); done
    else
        items+=("${condition}")
    fi
done
for item in "${items[@]}"; do
    condition="${item%%#*}"
    iteration_args=()
    it_tag=""
    if [[ "${item}" == *#* ]]; then
        iteration_args=(--student_iteration "${item#*#}")
        it_tag="-it${item#*#}"
    fi
    cycle="${condition%%@*}"
    selector_args=()
    if [[ "${condition}" == *@* ]]; then
        channel="${condition#*@}"
        case "${channel}" in
            h[0-9]*) selector_args=(--selector_hold "${channel#h}") ;;
            d[0-9]*) selector_args=(--selector_delay "${channel#d}") ;;
            only-reorient|only-turn) selector_args=(--only_expert "${channel#only-}") ;;
            *) echo "Unknown selector channel '${channel}' in '${condition}'"; overall_status=1; continue ;;
        esac
    fi
    label="${cycle//,/-}${channel:+-${channel}}${it_tag}"
    unset channel
    export SWITCH_CYCLE="${cycle}" HINGE_ASSET_DIR="${run_root}/asset-${label}"
    if [[ "${cycle}" == random ]]; then unset UNIFIED_CYCLE; else export UNIFIED_CYCLE="${cycle}"; fi
    result_json="${run_root}/${label}.json"
    run_log="${run_root}/${label}.log"
    echo "--- condition_start condition=${condition} cycle=${cycle} label=${label} ---"
    set +e
    timeout --signal=INT --kill-after=60s 900s \
        ./isaaclab.sh -p "${SLURM_SUBMIT_DIR}/evaluate_switch.py" \
        "${policy_args[@]}" ${selector_args[@]+"${selector_args[@]}"} ${iteration_args[@]+"${iteration_args[@]}"} \
        --num_envs "${num_envs}" --steps "${steps}" --seed "${seed}" \
        --output "${result_json}" ${SWITCH_CAPTURE:+--capture "${SWITCH_CAPTURE}" --capture_output "${run_root}/${label}-states.npz"} \
        --headless > "${run_log}" 2>&1
    command_status=$?
    set -e
    grep -E "^(captured_states|switch_cycle|student=|unified_goal_check|selector=|phases_completed|cycles_completed|falls_per|kind=|transition=|open_turn=|switch_evaluation_status)" "${run_log}" || true
    echo "command_status=${command_status}"
    cp "${run_log}" "${metadata_dir}/${label}.log"
    if [[ -f "${result_json}" ]]; then
        cp "${result_json}" "${metadata_dir}/${label}.json"
    else
        overall_status=1
        grep -vE "^\[|Warning" "${run_log}" | tail -n 40
    fi
    echo "--- condition_end condition=${condition} ---"
done

if [[ "${overall_status}" -eq 0 ]]; then
    echo "switch_eval_status=PASS"
else
    echo "switch_eval_status=FAIL"
    exit 1
fi
