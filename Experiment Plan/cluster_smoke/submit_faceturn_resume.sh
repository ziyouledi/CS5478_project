#!/usr/bin/env bash
# Face-turn student grid (FACE_TURN_PLAN.md E1) resumed 5000 -> 10000 iterations: idempotent, run repeatedly.
# Task i of the original grid (1153680_0-19, 1153770_20) is resumed from its iteration-5000 checkpoint as array
# task i of 25_train_and_eval.sbatch. Tasks whose checkpoint does not exist yet are skipped and picked up by a
# later call. State: faceturn_resume_state.txt, one line "<task> <job id> <checkpoint>" per submitted task.
# Usage: submit_faceturn_resume.sh [submit]
set -eo pipefail
cd ~/cluster_smoke
mode="${1:-dry}"
state=faceturn_resume_state.txt; touch "${state}"

export ISAACLAB_DIR="$HOME/software/isaaclab-stack/IsaacLab" VENV_DIR="$HOME/software/isaaclab-stack/env_isaaclab"
export INTENT_PCA_DIR="$HOME/data/low-dim-intent/datasets/action-pca-expert-faceturn-rollout-seed0-job1153675-job1153676"
export STUDENT_STAGE=faceturn-hidden EVAL_TASK=Isaac-FaceTurn-HingedCube-Shadow-Direct-v0
export STUDENT_INTENTS="blind:1 goal-yaw:1 goal-yaw:6 goal-yaw:30 hand-8:1 hand-8:30 hand-8:6"
unset STUDENT_RESUME_FROM

resume=(); ready=()
for i in $(seq 0 20); do
  name=$(STUDENT_ITERATIONS=5000 SLURM_ARRAY_TASK_ID=$i STUDENT_PRINT_PLAN=1 bash 16_train_student.sbatch 2>/dev/null | tail -1 | sed -n 's/.* name=//p')
  [[ -n "${name}" ]] || { echo "could not derive run name for task $i" >&2; exit 2; }
  ckpt=$(ls "$HOME"/data/low-dim-intent/runs/"${name}"-job{1153680,1153770}_"${i}"/"${name}"/nn/last_*_ep_5000_rew_*.pth 2>/dev/null | head -1 || true)
  if [[ -n "${ckpt}" ]]; then
    resume+=("${ckpt}")
    grep -q "^${i} " "${state}" || ready+=("${i}")
  else
    resume+=("MISSING_${i}")
  fi
done
export STUDENT_ITERATIONS=10000 STUDENT_RESUME_FROM="${resume[*]}"
echo "submitted_before=$(wc -l < "${state}") ready_now=${ready[*]:-none}"
(( ${#ready[@]} > 0 )) || exit 0
for i in "${ready[@]}"; do
  SLURM_ARRAY_TASK_ID=$i STUDENT_PRINT_PLAN=1 bash 16_train_student.sbatch 2>/dev/null | tail -1
done
q=$(squeue -u "$USER" -r -h < /dev/null | wc -l)
(( q + ${#ready[@]} <= 32 )) || { echo "NO_ROOM queued=${q}"; exit 0; }
[[ "${mode}" == "submit" ]] || { echo "DRY (pass 'submit')"; exit 0; }
array=$(IFS=,; echo "${ready[*]}")
TJ=$(sbatch --parsable --partition=gpu,gpu-long --time=03:00:00 --array="${array}" --export=ALL 25_train_and_eval.sbatch < /dev/null)
for i in "${ready[@]}"; do echo "${i} ${TJ} ${resume[$i]}" >> "${state}"; done
echo "faceturn_resume_submitted=${TJ} tasks=${array}"
