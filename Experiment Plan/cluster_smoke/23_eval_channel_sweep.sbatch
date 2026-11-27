#!/usr/bin/env bash
#SBATCH --job-name=shadow-channel-sweep
#SBATCH --partition=gpu,gpu-long
#SBATCH --gpus=1
#SBATCH --constraint=xgpf
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Rate sweep phase 2 P1: many last-checkpoint evaluations in one job (one GPU, one queue slot).
# Runs 17_evaluate_students.sbatch sequentially, once per (condition, run directory), with EVAL_LAST_ONLY=1.
# EVAL_RUN_DIRS: space-separated experiment directories (as for 17).
# CHANNEL_PLAN: space-separated entries <condition>@<i,j,...>, indices into EVAL_RUN_DIRS;
#   condition = base | delay<D> | bits<B> | hold<N>   (base = the run's trained channel;
#   delay0 forces no delay on a run trained under delay; hold<N> overrides the intent hold)
#   e.g. "base@0,1,2 delay6@0,1,2 bits2@0,1,2".
# Each evaluation writes its usual output/metadata directories (job tag <this job>_<index>).
# The job fails if any evaluation fails; all evaluations are attempted.

set -uo pipefail

: "${EVAL_RUN_DIRS:?Submit with EVAL_RUN_DIRS}"
: "${CHANNEL_PLAN:?Submit with CHANNEL_PLAN}"
cd "${SLURM_SUBMIT_DIR}"
test -f 17_evaluate_students.sbatch || exit 2
read -r -a run_dirs <<< "${EVAL_RUN_DIRS}"
read -r -a plan <<< "${CHANNEL_PLAN}"

echo "record_type=shadow_channel_sweep_job"
echo "job_id=${SLURM_JOB_ID}"
echo "channel_plan=${CHANNEL_PLAN}"

# Validate the whole plan before running anything.
for entry in "${plan[@]}"; do
    cond="${entry%@*}"; idx="${entry#*@}"
    if ! [[ "${entry}" == *@* && "${cond}" =~ ^(base|delay(0|[1-9][0-9]*)|bits[1-9][0-9]*|hold[1-9][0-9]*)$ && "${idx}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        echo "invalid plan entry '${entry}'" >&2; exit 2
    fi
    IFS=, read -r -a ids <<< "${idx}"
    for i in "${ids[@]}"; do
        (( i < ${#run_dirs[@]} )) || { echo "index ${i} outside ${#run_dirs[@]} run directories" >&2; exit 2; }
    done
done

n_pass=0; n_fail=0
for entry in "${plan[@]}"; do
    cond="${entry%@*}"; IFS=, read -r -a ids <<< "${entry#*@}"
    delay=""; bits=""; hold=""
    case "${cond}" in
        delay*) delay="${cond#delay}" ;;
        bits*) bits="${cond#bits}" ;;
        hold*) hold="${cond#hold}" ;;
    esac
    for i in "${ids[@]}"; do
        echo "=== channel_eval condition=${cond} index=${i} run_dir=${run_dirs[${i}]}"
        env -u SLURM_ARRAY_JOB_ID -u EVAL_INTENT_HOLD SLURM_ARRAY_TASK_ID="${i}" EVAL_LAST_ONLY=1 \
            EVAL_INTENT_DELAY="${delay}" EVAL_INTENT_BITS="${bits}" EVAL_INTENT_HOLD="${hold}" \
            bash 17_evaluate_students.sbatch
        st=$?
        if [[ "${st}" -eq 0 ]]; then
            n_pass=$((n_pass + 1)); echo "channel_eval_result condition=${cond} index=${i} status=PASS"
        else
            n_fail=$((n_fail + 1)); echo "channel_eval_result condition=${cond} index=${i} status=FAIL(${st})"
        fi
    done
done

echo "channel_evals_pass=${n_pass} channel_evals_fail=${n_fail}"
if (( n_fail == 0 )); then
    echo "channel_sweep_job_status=PASS"
else
    echo "channel_sweep_job_status=FAIL"; exit 1
fi
