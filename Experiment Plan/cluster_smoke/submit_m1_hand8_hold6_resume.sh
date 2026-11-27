#!/usr/bin/env bash
# M1 hand-8 at 10 Hz (intent hold 6) resumed 5000 -> 10000 iterations, seeds 0-4 (user request 2026-11-18):
# does the 10 Hz collapse persist with twice the training? Sources: newest ep_5000 checkpoint of
# 1111584_6-8 (seeds 0-2) and 1123488_8-9 (seeds 3-4). Array task i = seed i of 25_train_and_eval.sbatch
# (training + Stage 1 evaluation of checkpoints 5200-10000).
# Usage: submit_m1_hand8_hold6_resume.sh [submit]
set -eo pipefail
cd ~/cluster_smoke
mode="${1:-dry}"
runs="$HOME/data/low-dim-intent/runs"

export ISAACLAB_DIR="$HOME/software/isaaclab-stack/IsaacLab" VENV_DIR="$HOME/software/isaaclab-stack/env_isaaclab"
export INTENT_PCA_DIR="$HOME/data/low-dim-intent/datasets/action-pca-expert-rollout-seed0-job1023537-job1024096"
export STUDENT_STAGE=m1 STUDENT_INTENTS="hand-8:6" STUDENT_SEEDS="0 1 2 3 4"
unset STUDENT_RESUME_FROM STUDENT_ACTION_HOLD STUDENT_INTENT_HOLD EVAL_TASK

src=(job1111584_6 job1111584_7 job1111584_8 job1123488_8 job1123488_9)
resume=()
for s in 0 1 2 3 4; do
  name="student-m1-hand-8-hold6-seed${s}"
  ckpt=$(ls -t "${runs}/${name}-${src[$s]}/${name}"/nn/last_*_ep_5000_rew_*.pth 2>/dev/null | head -1 || true)
  [[ -n "${ckpt}" ]] || { echo "missing ep_5000 checkpoint for seed ${s}" >&2; exit 2; }
  resume+=("${ckpt}")
done
export STUDENT_ITERATIONS=10000 STUDENT_RESUME_FROM="${resume[*]}"
for i in 0 1 2 3 4; do
  echo "task ${i} <- ${resume[$i]}"
  SLURM_ARRAY_TASK_ID=$i STUDENT_PRINT_PLAN=1 bash 16_train_student.sbatch 2>/dev/null | tail -1
done
q=$(squeue -u "$USER" -r -h < /dev/null | wc -l)
(( q + 5 <= 32 )) || { echo "NO_ROOM queued=${q}"; exit 0; }
[[ "${mode}" == "submit" ]] || { echo "DRY (pass 'submit')"; exit 0; }
TJ=$(sbatch --parsable --partition=gpu,gpu-long --time=03:00:00 --array=0-4 --export=ALL 25_train_and_eval.sbatch < /dev/null)
echo "m1_hand8_hold6_resume_submitted=${TJ}"
