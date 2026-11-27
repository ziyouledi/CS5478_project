#!/usr/bin/env bash
# E3 `any` failure, diagnostics A and B (FACE_TURN_PLAN.md E3; user approved 2026-11-19). Run on the login node
# from ~/cluster_smoke after syncing intent_modules.py, train_student.py, evaluate_switch.py, 31_switch_eval.sbatch.
#   A  frozen v4 teacher, evaluation only: any / up / up,turn plain (must reproduce job 1159062), then obs[68:72]
#      (relative rotation), obs[64:68] (goal) or obs[64:72] replaced by another env's columns each step.
#   B  goal-11 student (goal-7 + relative rotation obs[68:72]), v4 training env (random goals, as the grid),
#      hold 1, seeds 0-1, 3000 iterations; evaluated like V1/V2 (any up up,turn random).
set -euo pipefail
cd ~/cluster_smoke
export ISAACLAB_DIR=$HOME/software/isaaclab-stack/IsaacLab VENV_DIR=$HOME/software/isaaclab-stack/env_isaaclab
export HINGE_REFERENCE_JSON=$HOME/data/low-dim-intent/runs/hinged-cube-eval-job1141499/dexcube.json
teacher=$HOME/data/low-dim-intent/checkpoints/expert-unified-cap0.05_hold10_v4-seed0/frozen/expert-unified-v4-seed0.pth
test "$(sha256sum "$teacher" | cut -d' ' -f1)" = ca3d9367a648a933a927e612697e2aa3fa37e66d564cff76c6ea7c8e250ca1d4
: > e3_relrot_jobs.txt

# A: teacher env = v4 training env with the evaluation protocol of 1159062 (no phase timeout, no start states)
A=$(
    SLURM_SUBMIT_DIR=$PWD
    source ./unified_env.sh
    unset UNIFIED_START_STATES UNIFIED_CYCLE E3_DIAG
    export UNIFIED_MAX_PHASE_STEPS=0 UNIFIED_CHECKPOINT=$teacher
    export SWITCH_CONDITIONS="any up up,turn any@mask-relrot up@mask-relrot up,turn@mask-relrot any@mask-goal up@mask-goal any@mask-both up@mask-both"
    sbatch --parsable --export=ALL --job-name=e3-teacher-mask --time=01:30:00 31_switch_eval.sbatch
)
echo "A teacher_mask eval=$A" | tee -a e3_relrot_jobs.txt

# B: goal-11 student, same settings as V1/V2 except the intent and the env (plain v4 training env)
export STUDENT_STAGE=unified-hidden STUDENT_ITERATIONS=3000 STUDENT_INTENTS="goal-11:1" STUDENT_SEEDS="0 1"
export INTENT_PCA_DIR=$HOME/data/low-dim-intent/datasets/action-pca-expert-unified-v4-rollout-seed0-job1160549-job1160550
unset UNIFIED_START_STATES UNIFIED_CYCLE UNIFIED_MAX_PHASE_STEPS E3_DIAG UNIFIED_CHECKPOINT
for i in 0 1; do SLURM_SUBMIT_DIR=$PWD SLURM_ARRAY_TASK_ID=$i STUDENT_PRINT_PLAN=1 bash 16_train_student.sbatch | tail -1; done
J=$(sbatch --parsable --export=ALL --job-name=e3-goal11 --array=0-1 16_train_student.sbatch)
dirs=""
for i in 0 1; do
    n=$(SLURM_SUBMIT_DIR=$PWD SLURM_ARRAY_TASK_ID=$i STUDENT_PRINT_PLAN=1 bash 16_train_student.sbatch | tail -1 | sed -n "s/.* name=//p")
    test -n "$n"
    dirs="$dirs $HOME/data/low-dim-intent/runs/$n-job${J}_$i/$n"
done
E=$(STUDENT_RUN_DIRS="${dirs# }" SWITCH_CONDITIONS="any up up,turn random" sbatch --parsable --export=ALL \
    --job-name=e3-goal11-eval --array=0-1 --output=logs/%x-%A_%a.out --error=logs/%x-%A_%a.err \
    --dependency=aftercorr:$J 31_switch_eval.sbatch)
echo "B goal-11 train=$J eval=$E" | tee -a e3_relrot_jobs.txt
printf "%s\n" $dirs >> e3_relrot_jobs.txt
sha256sum intent_modules.py train_student.py evaluate_switch.py 31_switch_eval.sbatch 16_train_student.sbatch unified_env.sh | cut -c1-12
sleep 3; squeue -u "$USER" -o "%.16i %.18j %.8T %R"
