#!/usr/bin/env bash
#SBATCH --job-name=faceturn-eval
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=xgpf
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Deterministic evaluation of face-turn experts (hinged_cube_env.py: ShadowHandFaceTurnEnv).
# FACE_TURN_EVALS: space-separated "label:effort_limit:checkpoint" entries.
# FACE_TURN_HOLD_STEPS (default 1) applies to every entry of the job.

set -Eeuo pipefail

: "${ISAACLAB_DIR:?Submit with ISAACLAB_DIR and VENV_DIR}"
: "${VENV_DIR:?Submit with ISAACLAB_DIR and VENV_DIR}"
: "${HINGE_REFERENCE_JSON:?Submit with HINGE_REFERENCE_JSON}"
: "${FACE_TURN_EVALS:?Submit with FACE_TURN_EVALS}"

source "${VENV_DIR}/bin/activate"
cd "${ISAACLAB_DIR}"

export PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA=YES
export XDG_CACHE_HOME="${HOME}/data/low-dim-intent/cache/xdg"

num_envs="${EVAL_NUM_ENVS:-256}"
episodes="${EVAL_EPISODES:-1024}"
seed="${EVAL_SEED:-20261006}"
run_root="${HOME}/data/low-dim-intent/runs/faceturn-eval-job${SLURM_JOB_ID}"
metadata_dir="${HOME}/data/low-dim-intent/metadata/faceturn-eval-job${SLURM_JOB_ID}"
mkdir -p "${run_root}" "${metadata_dir}" "${XDG_CACHE_HOME}"
for file in evaluate_shadow_checkpoint.py hinged_cube_env.py hinged_cube_tasks.py 30_eval_face_turn.sbatch; do
    cp "${SLURM_SUBMIT_DIR}/${file}" "${metadata_dir}/${file}"
done
scontrol show job "${SLURM_JOB_ID}" > "${metadata_dir}/slurm-job.txt"

echo "record_type=shadow_face_turn_expert_eval"
echo "timestamp=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname -f)"
echo "isaaclab_commit=$(git rev-parse HEAD)"
echo "num_envs=${num_envs}"
echo "episodes=${episodes}"
echo "evaluation_seed=${seed}"
echo "face_turn_hold_steps=${FACE_TURN_HOLD_STEPS:-1}"
echo "run_root=${run_root}"

overall_status=0
for entry in ${FACE_TURN_EVALS}; do
    IFS=: read -r label effort_limit checkpoint <<< "${entry}"
    export HINGE_EFFORT_LIMIT="${effort_limit}" HINGE_ASSET_DIR="${run_root}/asset-${label}"
    result_json="${run_root}/${label}.json"
    run_log="${run_root}/${label}.log"
    echo "--- evaluation_start label=${label} effort_limit=${effort_limit} ---"
    echo "checkpoint=${checkpoint}"
    echo "checkpoint_sha256=$(sha256sum "${checkpoint}" | cut -d' ' -f1)"
    set +e
    timeout --signal=INT --kill-after=60s 900s \
        ./isaaclab.sh -p "${SLURM_SUBMIT_DIR}/evaluate_shadow_checkpoint.py" \
        --task Isaac-FaceTurn-HingedCube-Shadow-Direct-v0 \
        --checkpoint "${checkpoint}" \
        --num_envs "${num_envs}" --episodes "${episodes}" --seed "${seed}" \
        --output "${result_json}" --headless > "${run_log}" 2>&1
    command_status=$?
    set -e
    grep -E "^(hinged_cube |face_turn_hold_steps|mean_episode_reward|std_episode_reward|mean_episode_length|mean_consecutive_successes|hinge=|checkpoint_evaluation_status)" "${run_log}" || true
    echo "evaluation_command_status=${command_status}"
    cp "${run_log}" "${metadata_dir}/${label}.log"
    if [[ -f "${result_json}" ]]; then
        cp "${result_json}" "${metadata_dir}/${label}.json"
    else
        overall_status=1
        tail -n 40 "${run_log}"
    fi
    echo "--- evaluation_end label=${label} ---"
done

if [[ "${overall_status}" -eq 0 ]]; then
    echo "face_turn_eval_status=PASS"
else
    echo "face_turn_eval_status=FAIL"
    exit 1
fi
