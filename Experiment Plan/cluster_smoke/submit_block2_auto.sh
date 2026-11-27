#!/usr/bin/env bash
# Rate sweep phase 2, block 2 (goal-4 m1 resumed 5000 -> 10000): idempotent auto-submission, run repeatedly.
# 1. Training (once): when the queue has room for 9 tasks, submit 16 with --array=0-8%7 --nice=100 (user decision
#    2026-11-12: use up to the cluster's 8 GPUs per user; training at lower priority than evaluations, so a freed GPU
#    goes to a waiting evaluation; the QOS caps the total at 8). Job ID -> phase2_block2_train_job.txt.
# 2. Evaluations: only after all block 1 evaluations are submitted (BLOCK1_EVAL_STATE has BLOCK1_N lines);
#    via submit_block_evals.sh, chained afterany on the last block 1 evaluation array while it is queued.
# Inputs (written beforehand): phase2_block2_intents.txt, phase2_block2_resume_from.txt (9 checkpoints, task order).
# Usage: submit_block2_auto.sh BLOCK1_TRAIN_JOB [submit]
set -eo pipefail
cd ~/cluster_smoke
block1_job="$1"; mode="${2:-dry}"
block1_state="eval_state_${block1_job}.txt"; block1_n="${BLOCK1_N:-27}"
max_queue="${MAX_QUEUE:-32}"
job_file=phase2_block2_train_job.txt

export ISAACLAB_DIR="$HOME/software/isaaclab-stack/IsaacLab" VENV_DIR="$HOME/software/isaaclab-stack/env_isaaclab"
export INTENT_PCA_DIR="$HOME/data/low-dim-intent/datasets/action-pca-expert-rollout-seed0-job1023537-job1024096"
export STUDENT_ITERATIONS=10000 STUDENT_STAGE=m1
export STUDENT_INTENTS="$(cat phase2_block2_intents.txt)" STUDENT_RESUME_FROM="$(cat phase2_block2_resume_from.txt)"
read -r -a resume <<< "${STUDENT_RESUME_FROM}"
(( ${#resume[@]} == 9 )) || { echo "expected 9 resume checkpoints, got ${#resume[@]}" >&2; exit 2; }
for f in "${resume[@]}"; do test -f "$f" || { echo "missing $f" >&2; exit 2; }; done

q=$(squeue -u "$USER" -r -h < /dev/null | wc -l)
if [[ ! -s "${job_file}" ]]; then
  echo "block2_training=unsubmitted queued=${q}"
  (( q + 9 <= max_queue )) || { echo "NO_ROOM_TRAINING"; exit 0; }
  [[ "${mode}" == "submit" ]] || { echo "DRY (pass 'submit')"; exit 0; }
  TJ=$(sbatch --parsable --array=0-8%7 --nice=100 --export=ALL 16_train_student.sbatch < /dev/null)
  echo "${TJ}" > "${job_file}"
  # Run dirs: runs/<name>-job<TJ>_<i>/<name>, name from the print-plan mapping.
  dirs=()
  for i in $(seq 0 8); do
    name=$(SLURM_ARRAY_TASK_ID=$i STUDENT_PRINT_PLAN=1 bash 16_train_student.sbatch 2>/dev/null | tail -1 | sed -n 's/.* name=//p')
    [[ -n "${name}" ]] || { echo "could not derive run name for task $i" >&2; exit 2; }
    dirs+=("$HOME/data/low-dim-intent/runs/${name}-job${TJ}_${i}/${name}")
  done
  echo "${dirs[*]}" > "phase2_block2_eval_dirs_${TJ}.txt"
  echo "block2_training_submitted=${TJ}"
  q=$(squeue -u "$USER" -r -h < /dev/null | wc -l)
fi
TJ=$(cat "${job_file}")
echo "block2_training=${TJ}"

b1=$(wc -l < "${block1_state}" 2>/dev/null || echo 0)
if (( b1 < block1_n )); then echo "block1_evals_submitted=${b1}/${block1_n}; block 2 evaluations wait"; exit 0; fi
b2=$(wc -l < "eval_state_${TJ}.txt" 2>/dev/null || echo 0)
if (( b2 >= 9 )); then echo "block2_evals_submitted=9/9 BLOCK2_DONE"; exit 0; fi
last1=$(awk '{print $2}' "${block1_state}" | tail -1)
extra=""
if [[ -n "${last1}" ]] && squeue -h -j "${last1}" < /dev/null 2>/dev/null | grep -q .; then extra="afterany:${last1}"; fi
EXTRA_DEP="${extra}" ./submit_block_evals.sh "${TJ}" "phase2_block2_eval_dirs_${TJ}.txt" 0 8 \
  Isaac-Repose-Cube-Shadow-Direct-v0 "${mode}"
echo "block2_evals_submitted=$(wc -l < "eval_state_${TJ}.txt" 2>/dev/null || echo 0)/9"
