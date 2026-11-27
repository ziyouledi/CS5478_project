#!/usr/bin/env bash
#SBATCH --job-name=shadow-hinged-cube
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=xgpf
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Evaluate the frozen Stage 1 expert, without retraining, on a two-part hinged cube that has the
# DexCube's size and mass. First run: the stock task (reference; also measures the DexCube).
# Then one run per hinge setting "label:stiffness:damping:effort_limit".

set -Eeuo pipefail

: "${ISAACLAB_DIR:?Submit with ISAACLAB_DIR and VENV_DIR}"
: "${VENV_DIR:?Submit with ISAACLAB_DIR and VENV_DIR}"

source "${VENV_DIR}/bin/activate"
cd "${ISAACLAB_DIR}"

export PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA=YES
export XDG_CACHE_HOME="${HOME}/data/low-dim-intent/cache/xdg"

checkpoint="${CHECKPOINT:-${HOME}/data/low-dim-intent/checkpoints/expert-seed0/frozen/expert-seed0.pth}"
num_envs="${EVAL_NUM_ENVS:-256}"
episodes="${EVAL_EPISODES:-1024}"
seed="${EVAL_SEED:-20261006}"
conditions="${HINGE_CONDITIONS:-locked:1000:1:100 cap0.3:5:0.02:0.3 cap0.1:5:0.02:0.1 cap0.03:5:0.02:0.03 cap0.01:5:0.02:0.01 free:0:0:100}"
run_root="${HOME}/data/low-dim-intent/runs/hinged-cube-eval-job${SLURM_JOB_ID}"
metadata_dir="${HOME}/data/low-dim-intent/metadata/hinged-cube-eval-job${SLURM_JOB_ID}"
mkdir -p "${run_root}" "${metadata_dir}" "${XDG_CACHE_HOME}"

evaluator="${SLURM_SUBMIT_DIR}/evaluate_shadow_checkpoint.py"
for file in evaluate_shadow_checkpoint.py hinged_cube_env.py hinged_cube_tasks.py 29_hinged_cube_eval.sbatch; do
    test -f "${SLURM_SUBMIT_DIR}/${file}"
    cp "${SLURM_SUBMIT_DIR}/${file}" "${metadata_dir}/${file}"
done
test -f "${checkpoint}"
scontrol show job "${SLURM_JOB_ID}" > "${metadata_dir}/slurm-job.txt"

echo "record_type=shadow_hinged_cube_expert_eval"
echo "timestamp=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname -f)"
echo "isaaclab_commit=$(git rev-parse HEAD)"
echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd ';')"
echo "checkpoint=${checkpoint}"
echo "checkpoint_sha256=$(sha256sum "${checkpoint}" | cut -d' ' -f1)"
echo "num_envs=${num_envs}"
echo "episodes=${episodes}"
echo "evaluation_seed=${seed}"
echo "conditions=${conditions}"
echo "run_root=${run_root}"

run_eval() {
    local label="$1" task="$2"
    local result_json="${run_root}/${label}.json"
    local run_log="${run_root}/${label}.log"
    echo "--- evaluation_start label=${label} task=${task} ---"
    local start_epoch
    start_epoch=$(date +%s)
    set +e
    timeout --signal=INT --kill-after=60s 900s \
        ./isaaclab.sh -p "${evaluator}" \
        --task "${task}" \
        --checkpoint "${checkpoint}" \
        --num_envs "${num_envs}" \
        --episodes "${episodes}" \
        --seed "${seed}" \
        --output "${result_json}" \
        --headless \
        > "${run_log}" 2>&1
    local command_status=$?
    set -e
    grep -E "^(object_probe|hinged_cube|mean_episode_reward|std_episode_reward|mean_episode_length|mean_consecutive_successes|final_consecutive_successes|checkpoint_evaluation_status)" "${run_log}" || true
    echo "evaluation_label=${label}"
    echo "evaluation_elapsed_seconds=$(($(date +%s) - start_epoch))"
    echo "evaluation_command_status=${command_status}"
    cp "${run_log}" "${metadata_dir}/${label}.log"
    if [[ -f "${result_json}" ]]; then
        cp "${result_json}" "${metadata_dir}/${label}.json"
    else
        command_status=1
        tail -n 40 "${run_log}"
    fi
    echo "--- evaluation_end label=${label} ---"
    return "${command_status}"
}

overall_status=0
if ! run_eval dexcube Isaac-Repose-Cube-Shadow-Direct-v0; then
    echo "hinged_cube_eval_status=FAIL (reference run failed)"
    exit 1
fi

export HINGE_REFERENCE_JSON="${run_root}/dexcube.json"
for condition in ${conditions}; do
    IFS=: read -r label stiffness damping effort_limit <<< "${condition}"
    export HINGE_STIFFNESS="${stiffness}" HINGE_DAMPING="${damping}" HINGE_EFFORT_LIMIT="${effort_limit}"
    export HINGE_ASSET_DIR="${run_root}/asset-${label}"
    if ! run_eval "hinge-${label}" Isaac-Repose-HingedCube-Shadow-Direct-v0; then
        overall_status=1
    fi
done

if [[ "${overall_status}" -eq 0 ]]; then
    echo "hinged_cube_eval_status=PASS"
else
    echo "hinged_cube_eval_status=FAIL"
    exit 1
fi
