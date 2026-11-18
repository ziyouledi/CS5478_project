#!/usr/bin/env bash
#SBATCH --job-name=shadow-student-dry
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=xgpf
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# End-to-end dry run: every stage-one intent trains a student for a few PPO iterations with a
# small environment count. Checks wiring (shapes, expert on GPU, logging), not learning.
# DRY_RUN_HOLDS (default "1") lists intent holds; every intent runs once per hold (rate sweep).
# DRY_RUN_DELAY=d (default 0, rate sweep phase 3) trains every run under an intent delay of d steps (use holds "1").
# Timing (E3): DRY_RUN_NUM_ENVS=2048 DRY_RUN_ITERATIONS=60 prints dry_run_timing with the steady fps and the
# estimated wall time of DRY_RUN_PLAN_ITERATIONS (default 5000) per intent / hold.
# DRY_RUN_RESUME_FROM=<periodic checkpoint> (rate sweep phase 2) resumes every run from that checkpoint for
# DRY_RUN_ITERATIONS more epochs; set DRY_RUN_INTENTS / DRY_RUN_HOLDS / STUDENT_STAGE to match its run (seed 0).

set -Eeuo pipefail

: "${ISAACLAB_DIR:?Submit with ISAACLAB_DIR and VENV_DIR}"
: "${VENV_DIR:?Submit with ISAACLAB_DIR and VENV_DIR}"
: "${INTENT_PCA_DIR:?Submit with INTENT_PCA_DIR (a completed action-pca output directory)}"
: "${OBJ_LOOKAHEAD:?Submit with OBJ_LOOKAHEAD (radians)}"

source "${VENV_DIR}/bin/activate"
cd "${ISAACLAB_DIR}"

export PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA=YES
export XDG_CACHE_HOME="${HOME}/data/low-dim-intent/cache/xdg"

intents="${DRY_RUN_INTENTS:-blind wrist-2 hand-2 hand-4 hand-20 obj-1 obj-4}"
num_envs="${DRY_RUN_NUM_ENVS:-256}"
iterations="${DRY_RUN_ITERATIONS:-3}"
holds="${DRY_RUN_HOLDS:-1}"
delay="${DRY_RUN_DELAY:-0}"
action_hold="${DRY_RUN_ACTION_HOLD:-1}"
seed=0
horizon_length=16
minibatch_size=$((num_envs * horizon_length))
stage="${STUDENT_STAGE:-stage1}"
# STUDENT_STAGE=m2 trains on the M2 task (shadow_zaxis_env.py) with the frozen M2 expert.
# STUDENT_STAGE=m1 is Stage 2 M1: Stage 1 task and expert, goal hidden from the actor, asymmetric
# critic. STUDENT_STAGE=m1-gate keeps the goal visible with the same asymmetric critic (M1 gate run).
# STUDENT_STAGE=m2-hidden is M1's setting on the M2 task and expert (rate sweep phase 2).
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
frozen_dir="${HOME}/data/low-dim-intent/checkpoints/${expert_name}/frozen"
expert_file="${expert_file:-${expert_name}.pth}"
checkpoint="${frozen_dir}/${expert_file}"
expert_sha256=$(awk -v name="${expert_file}" '$2 == name {print $1}' "${frozen_dir}/SHA256SUMS")
pca_basis="${INTENT_PCA_DIR}/pca_basis.npz"
# The action PCA must come from the same task's expert rollout (M2 basis directories contain "expert-m2").
case "${stage}" in
    m2*) [[ "${INTENT_PCA_DIR}" == *expert-m2* ]] || { echo "STUDENT_STAGE=${stage} needs the M2 action PCA" >&2; exit 2; } ;;
    faceturn*) [[ "${INTENT_PCA_DIR}" == *expert-faceturn* ]] || { echo "STUDENT_STAGE=${stage} needs the face-turn action PCA" >&2; exit 2; } ;;
    unified*) [[ "${INTENT_PCA_DIR}" == *expert-unified* ]] || { echo "STUDENT_STAGE=${stage} needs the unified v4 action PCA" >&2; exit 2; } ;;
    *) [[ "${INTENT_PCA_DIR}" != *expert-m2* && "${INTENT_PCA_DIR}" != *expert-faceturn* && "${INTENT_PCA_DIR}" != *expert-unified* ]] || { echo "STUDENT_STAGE=${stage} must not use the M2, face-turn or unified action PCA" >&2; exit 2; } ;;
esac
resume_args=()
resume_epoch=0
if [[ -n "${DRY_RUN_RESUME_FROM:-}" ]]; then
    test -f "${DRY_RUN_RESUME_FROM}"
    resume_epoch=$(basename "${DRY_RUN_RESUME_FROM}" | sed -n 's/.*_ep_\([0-9]*\)_.*/\1/p')
    test -n "${resume_epoch}"
    resume_args=(--checkpoint "${DRY_RUN_RESUME_FROM}")
fi
run_root="${HOME}/data/low-dim-intent/runs/student-dry-run-job${SLURM_JOB_ID}"
metadata_dir="${HOME}/data/low-dim-intent/metadata/student-dry-run-job${SLURM_JOB_ID}"
trainer="${SLURM_SUBMIT_DIR}/train_student.py"

test -f "${checkpoint}"
test -n "${expert_sha256}"
test -f "${pca_basis}"
if [[ -n "${JOINT_PCA_DIR:-}" ]]; then test -f "${JOINT_PCA_DIR}/joint_basis.npz"; fi
test -f "${trainer}"
mkdir -p "${run_root}" "${metadata_dir}" "${XDG_CACHE_HOME}"
cp "${SLURM_SUBMIT_DIR}"/{train_student.py,intent_env_wrapper.py,intent_modules.py,student_eval_utils.py,m2_tasks.py,shadow_zaxis_env.py,15_student_dry_run.sbatch} "${metadata_dir}/"
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

echo "record_type=shadow_student_dry_run_job"
echo "timestamp=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname -f)"
echo "isaaclab_commit=$(git rev-parse HEAD)"
echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd ';')"
echo "stage=${stage}"
echo "task=${task}"
echo "student_flags=${student_flags[*]:-none}"
echo "expert_checkpoint=${checkpoint}"
echo "joint_pca_dir=${JOINT_PCA_DIR:-none}"
echo "intents=${intents}"
echo "num_envs=${num_envs}"
echo "iterations=${iterations}"
echo "holds=${holds}"
echo "delay=${delay}"
echo "action_hold=${action_hold}"
echo "minibatch_size=${minibatch_size}"
echo "expert_sha256=${expert_sha256}"
echo "pca_basis=${pca_basis}"
echo "obj_lookahead=${OBJ_LOOKAHEAD}"
echo "resume_from=${DRY_RUN_RESUME_FROM:-none}"
echo "resume_start_epoch=${resume_epoch}"

failed=0
for hold in ${holds}; do
for intent in ${intents}; do
    name="dry-${intent}-hold${hold}-seed${seed}"
    log="${run_root}/${name}.log"
    case "${intent}" in
        joint-*) intent_basis="${JOINT_PCA_DIR:?joint-K intents need JOINT_PCA_DIR}/joint_basis.npz" ;;
        *) intent_basis="${pca_basis}" ;;
    esac
    start_epoch=$(date +%s)
    set +e
    timeout --signal=INT --kill-after=30s 600s \
        ./isaaclab.sh -p "${trainer}" \
        --task "${task}" \
        --intent "${intent}" \
        --expert_checkpoint "${checkpoint}" \
        --expert_sha256 "${expert_sha256}" \
        --pca_basis "${intent_basis}" \
        "${student_flags[@]}" \
        --obj_lookahead "${OBJ_LOOKAHEAD}" \
        --intent_hold "${hold}" \
        --intent_delay "${delay}" \
        --action_hold "${action_hold}" \
        "${resume_args[@]}" \
        --num_envs "${num_envs}" \
        --seed "${seed}" \
        --max_iterations "$((resume_epoch + iterations))" \
        --train_dir "${run_root}" \
        --experiment_name "${name}" \
        --headless \
        agent.params.config.minibatch_size="${minibatch_size}" \
        > "${log}" 2>&1
    status=$?
    set -e
    elapsed=$(( $(date +%s) - start_epoch ))
    shape=$(grep -m1 -o 'student_obs_shape=([0-9, ]*) intent_dim=[0-9]*' "${log}" || echo "student_obs_shape=missing")
    logged_hold=$(grep -m1 -o 'intent_hold=[0-9]*' "${log}" || echo "intent_hold=missing")
    if [[ "${logged_hold}" != "intent_hold=${hold}" ]]; then
        echo "logged ${logged_hold} does not match requested hold ${hold}" >&2
        status=3
    fi
    logged_delay=$(grep -m1 -o 'intent_delay=[0-9]*' "${log}" || echo "intent_delay=missing")
    if [[ "${logged_delay}" != "intent_delay=${delay}" ]]; then
        echo "logged ${logged_delay} does not match requested delay ${delay}" >&2
        status=5
    fi
    logged_ahold=$(grep -m1 -o 'action_hold=[0-9]*' "${log}" || echo "action_hold=missing")
    if [[ "${logged_ahold}" != "action_hold=${action_hold}" ]]; then
        echo "logged ${logged_ahold} does not match requested action hold ${action_hold}" >&2
        status=6
    fi
    epochs=$(grep -c "fps step:" "${log}" || true)
    if [[ -n "${DRY_RUN_RESUME_FROM:-}" ]]; then
        # train_student.py prints this line (start of line) after checking the restored weights.
        resume_line=$(grep -m1 '^resume_start_epoch=' "${log}" || echo "resume_start_epoch=missing")
        echo "dry_run_resume ${resume_line}"
        if [[ "${resume_line}" != "resume_start_epoch=${resume_epoch} "* ]]; then
            echo "resumed run did not report start epoch ${resume_epoch}" >&2
            status=4
        fi
    fi
    if [[ "${status}" -eq 0 ]] && grep -q "student_train_status=PASS" "${log}"; then
        result=PASS
    else
        result=FAIL
        failed=$((failed + 1))
        tail -40 "${log}" >&2
    fi
    echo "dry_run intent=${intent} hold=${hold} delay=${delay} action_hold=${action_hold} status=${result} exit=${status} elapsed_s=${elapsed} reported_epochs=${epochs} ${shape}"
    # Timing: steady "fps total" (first 5 epochs dropped) -> wall time of a full run of DRY_RUN_PLAN_ITERATIONS
    # (default 5000) at this env count. Only meaningful with DRY_RUN_NUM_ENVS = the grid's env count.
    python - "${log}" "${num_envs}" "${horizon_length}" "${DRY_RUN_PLAN_ITERATIONS:-5000}" <<'PY'
import re
import statistics
import sys

text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
num_envs, horizon, plan = (int(v) for v in sys.argv[2:5])
fps = [int(v) for v in re.findall(r"fps total:\s*(\d+)", text)]
steady = fps[5:]
if steady:
    median = statistics.median(steady)
    seconds = plan * num_envs * horizon / median
    print(f"dry_run_timing epochs={len(fps)} steady_fps_total_median={median:.0f} "
          f"estimated_train_s_{plan}it={seconds:.0f} estimated_train_h_{plan}it={seconds / 3600:.2f}")
else:
    print(f"dry_run_timing epochs={len(fps)} steady_fps_total_median=n/a (need more than 5 epochs)")
PY
    cp "${log}" "${metadata_dir}/"
    if [[ "${result}" == FAIL ]]; then
        echo "dry_run_stopped_after=${intent} hold=${hold}"
        break 2
    fi
done
done

echo "dry_run_failures=${failed}"
if [[ "${failed}" -eq 0 ]]; then
    echo "student_dry_run_job_status=PASS"
else
    echo "student_dry_run_job_status=FAIL"
    exit 1
fi
