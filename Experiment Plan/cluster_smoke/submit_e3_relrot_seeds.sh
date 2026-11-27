#!/usr/bin/env bash
# E3 `any` failure, more seeds (user 2026-11-19): goal-11 seeds 2-5 and goal-7 seeds 3-5, same settings as
# goal-11 job 1166088 (v4 training env, random goals, hold 1, 3000 iterations, 2048 envs); evaluated like 1166089.
# With goal-11 seeds 0-1 (1166088) and goal-7 seeds 0-2 (grid 1160744, 3000-iteration checkpoints in 1163578)
# this gives 6 seeds each. Rule: `any` >= 1 per 600 steps at iteration 3000 = learned.
set -euo pipefail
cd ~/cluster_smoke
export ISAACLAB_DIR=$HOME/software/isaaclab-stack/IsaacLab VENV_DIR=$HOME/software/isaaclab-stack/env_isaaclab
export HINGE_REFERENCE_JSON=$HOME/data/low-dim-intent/runs/hinged-cube-eval-job1141499/dexcube.json
export STUDENT_STAGE=unified-hidden STUDENT_ITERATIONS=3000
export INTENT_PCA_DIR=$HOME/data/low-dim-intent/datasets/action-pca-expert-unified-v4-rollout-seed0-job1160549-job1160550
unset UNIFIED_START_STATES UNIFIED_CYCLE UNIFIED_MAX_PHASE_STEPS E3_DIAG UNIFIED_CHECKPOINT
: > e3_relrot_seeds_jobs.txt
submit() {  # intent, seeds, count
    local intent=$1 seeds=$2 count=$3 J E dirs="" n i
    export STUDENT_INTENTS="${intent}:1" STUDENT_SEEDS="${seeds}"
    for i in $(seq 0 $((count - 1))); do SLURM_SUBMIT_DIR=$PWD SLURM_ARRAY_TASK_ID=$i STUDENT_PRINT_PLAN=1 bash 16_train_student.sbatch | tail -1; done
    J=$(sbatch --parsable --export=ALL --job-name=e3-${intent}-seeds --array=0-$((count - 1)) 16_train_student.sbatch)
    for i in $(seq 0 $((count - 1))); do
        n=$(SLURM_SUBMIT_DIR=$PWD SLURM_ARRAY_TASK_ID=$i STUDENT_PRINT_PLAN=1 bash 16_train_student.sbatch | tail -1 | sed -n "s/.* name=//p")
        test -n "$n"
        dirs="$dirs $HOME/data/low-dim-intent/runs/$n-job${J}_$i/$n"
    done
    E=$(STUDENT_RUN_DIRS="${dirs# }" SWITCH_CONDITIONS="any up up,turn random" sbatch --parsable --export=ALL \
        --job-name=e3-${intent}-seeds-eval --array=0-$((count - 1)) --output=logs/%x-%A_%a.out --error=logs/%x-%A_%a.err \
        --dependency=aftercorr:$J 31_switch_eval.sbatch)
    echo "${intent} seeds=${seeds} train=$J eval=$E" | tee -a e3_relrot_seeds_jobs.txt
    printf "%s\n" $dirs >> e3_relrot_seeds_jobs.txt
}
submit goal-11 "2 3 4 5" 4
submit goal-7 "3 4 5" 3
sha256sum intent_modules.py train_student.py evaluate_switch.py 31_switch_eval.sbatch 16_train_student.sbatch unified_env.sh | cut -c1-12
sleep 3; squeue -u "$USER" -o "%.16i %.22j %.8T %R"
