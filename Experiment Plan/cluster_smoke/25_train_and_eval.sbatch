#!/usr/bin/env bash
#SBATCH --job-name=shadow-train-eval
#SBATCH --partition=gpu-long
#SBATCH --gpus=1
#SBATCH --constraint=xgpf
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Train one student (16) and evaluate it (17) in the same job, so a run costs one queue entry instead of two.
# Takes the variables of 16_train_student.sbatch; EVAL_TASK as in 17 (default Stage 1).
set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR}"
task_id="${SLURM_ARRAY_TASK_ID:-0}"
name=$(STUDENT_PRINT_PLAN=1 bash 16_train_student.sbatch | tail -1 | sed -n 's/.* name=//p')
test -n "${name}"
run_dir="${HOME}/data/low-dim-intent/runs/${name}-job${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${task_id}/${name}"
echo "train_and_eval name=${name} run_dir=${run_dir}"
bash 16_train_student.sbatch
cd "${SLURM_SUBMIT_DIR}"
test -d "${run_dir}/nn"
# 17 indexes EVAL_RUN_DIRS by the array task id; give it this run as entry 0. Its output directory is
# eval-<name>-job<array job>_0 for every task, unique through <name>.
EVAL_RUN_DIRS="${run_dir}" SLURM_ARRAY_TASK_ID=0 bash 17_evaluate_students.sbatch
echo "train_and_eval_status=PASS"
