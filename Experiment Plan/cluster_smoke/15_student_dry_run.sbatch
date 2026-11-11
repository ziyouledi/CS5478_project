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
seed=0
horizon_length=16
minibatch_size=$((num_envs * horizon_length))
stage="${STUDENT_STAGE:-stage1}"
# STUDENT_STAGE=m2 trains on the M2 task (shadow_zaxis_env.py) with the frozen M2 expert.
# STUDENT_STAGE=m1 is Stage 2 M1: Stage 1 task and expert, goal hidden from the actor, asymmetric
# critic. STUDENT_STAGE=m1-gate keeps the goal visible with the same asymmetric critic (M1 gate run).
student_flags=()
case "${stage}" in
    m2) task=Isaac-Repose-Cube-Shadow-ZAxis-Direct-v0; expert_name=expert-m2-seed0 ;;
    m1) task=Isaac-Repose-Cube-Shadow-Direct-v0; expert_name=expert-seed0
        student_flags=(--goal_hidden --asymmetric_critic) ;;
    m1-gate) task=Isaac-Repose-Cube-Shadow-Direct-v0; expert_name=expert-seed0
        student_flags=(--asymmetric_critic) ;;
    stage1) task=Isaac-Repose-Cube-Shadow-Direct-v0; expert_name=expert-seed0 ;;
    *) echo "unknown STUDENT_STAGE=${stage}" >&2; exit 2 ;;
esac
frozen_dir="${HOME}/data/low-dim-intent/checkpoints/${expert_name}/frozen"
checkpoint="${frozen_dir}/${expert_name}.pth"
expert_sha256=$(awk -v name="${expert_name}.pth" '$2 == name {print $1}' "${frozen_dir}/SHA256SUMS")
pca_basis="${INTENT_PCA_DIR}/pca_basis.npz"
run_root="${HOME}/data/low-dim-intent/runs/student-dry-run-job${SLURM_JOB_ID}"
metadata_dir="${HOME}/data/low-dim-intent/metadata/student-dry-run-job${SLURM_JOB_ID}"
trainer="${SLURM_SUBMIT_DIR}/train_student.py"

test -f "${checkpoint}"
test -n "${expert_sha256}"
test -f "${pca_basis}"
if [[ -n "${JOINT_PCA_DIR:-}" ]]; then test -f "${JOINT_PCA_DIR}/joint_basis.npz"; fi
test -f "${trainer}"
mkdir -p "${run_root}" "${metadata_dir}" "${XDG_CACHE_HOME}"
cp "${SLURM_SUBMIT_DIR}"/{train_student.py,intent_env_wrapper.py,intent_modules.py,m2_tasks.py,shadow_zaxis_env.py,15_student_dry_run.sbatch} "${metadata_dir}/"
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
echo "minibatch_size=${minibatch_size}"
echo "expert_sha256=${expert_sha256}"
echo "pca_basis=${pca_basis}"
echo "obj_lookahead=${OBJ_LOOKAHEAD}"

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
        --num_envs "${num_envs}" \
        --seed "${seed}" \
        --max_iterations "${iterations}" \
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
    epochs=$(grep -c "fps step:" "${log}" || true)
    if [[ "${status}" -eq 0 ]] && grep -q "student_train_status=PASS" "${log}"; then
        result=PASS
    else
        result=FAIL
        failed=$((failed + 1))
        tail -40 "${log}" >&2
    fi
    echo "dry_run intent=${intent} hold=${hold} status=${result} exit=${status} elapsed_s=${elapsed} reported_epochs=${epochs} ${shape}"
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
