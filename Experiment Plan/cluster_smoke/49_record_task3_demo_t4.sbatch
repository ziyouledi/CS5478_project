#!/usr/bin/env bash
#SBATCH --job-name=task3-demo-t4
#SBATCH --partition=gpu,gpu-long
#SBATCH --gpus=1
#SBATCH --constraint=xgpf
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Records 4 clean single-hand (num_envs=1) high-camera 6-colour Rubik's-cube Task 3 clips:
#   1. Unified v4 expert (60 Hz, up,turn)
#   2. E3 hand-8 @ 60 Hz (seed 0)
#   3. E3 goal-7 @ 10 Hz (seed 1, hold=6)
#   4. E3 hand-8 @ 10 Hz (seed 2, hold=6)

set -Eeuo pipefail

ISAAC_VENV="${HOME}/software/isaaclab-stack/env_isaaclab"
ISAACLAB_DIR="${HOME}/software/isaaclab-stack/IsaacLab"
V4_CKPT="${HOME}/data/low-dim-intent/checkpoints/expert-unified-cap0.05_hold10_v4-seed0/frozen/expert-unified-v4-seed0.pth"
RUNS="${HOME}/data/low-dim-intent/runs"
OUT_DIR="${HOME}/data/low-dim-intent/videos/task3-unified-rubik-demo-job${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}" logs
cp "${SLURM_SUBMIT_DIR}"/{record_task3_unified_demo.py,49_record_task3_demo_t4.sbatch} "${OUT_DIR}/"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"
export OMNI_KIT_ACCEPT_EULA=YES
export XDG_CACHE_HOME="${HOME}/data/low-dim-intent/cache/xdg"
export HINGE_ASSET_DIR="${OUT_DIR}/asset"
export HINGE_EFFORT_LIMIT=0.05
export FACE_TURN_HOLD_STEPS=10
export HINGE_REFERENCE_JSON="${HOME}/data/low-dim-intent/runs/hinged-cube-eval-job1141499/dexcube.json"
export UNIFIED_EXTRA_OBS=1
export UNIFIED_CYCLE="up,turn"
# High-angle single-hand framing matching Task 1 & Task 2 clips
export Z1_CAMERAS="high:-50:48:0.60"
export Z1_IMAGE="600" Z1_FOCAL="24" Z1_TARGET="0 -0.39 0.6"
mkdir -p "${XDG_CACHE_HOME}" "${HINGE_ASSET_DIR}"
unset UNIFIED_START_STATES FACE_TURN_START_STATES HINGE_STIFFNESS HINGE_DAMPING

echo "record_type=task3_demo_t4 job_id=${SLURM_JOB_ID} node=$(hostname -f) out_dir=${OUT_DIR}"

cd "${ISAACLAB_DIR}"

echo "=== 1/4: Unified v4 (60 Hz) ==="
"${ISAAC_VENV}/bin/python" "${SLURM_SUBMIT_DIR}/record_task3_unified_demo.py" \
    --headless --unified_checkpoint "${V4_CKPT}" \
    --name "task3_unified_v4_60hz" --steps 600 --output_dir "${OUT_DIR}"

echo "=== 2/4: E3 hand-8 (60 Hz, seed 0) ==="
"${ISAAC_VENV}/bin/python" "${SLURM_SUBMIT_DIR}/record_task3_unified_demo.py" \
    --headless --student_run_dir "${RUNS}/student-unified-hidden-hand-8-seed0-job1160744_9/student-unified-hidden-hand-8-seed0" \
    --name "task3_e3_hand8_60hz" --steps 600 --output_dir "${OUT_DIR}"

echo "=== 3/4: E3 goal-7 (10 Hz, hold=6, seed 1) ==="
"${ISAAC_VENV}/bin/python" "${SLURM_SUBMIT_DIR}/record_task3_unified_demo.py" \
    --headless --student_run_dir "${RUNS}/student-unified-hidden-goal-7-hold6-seed1-job1160744_7/student-unified-hidden-goal-7-hold6-seed1" \
    --name "task3_e3_goal7_10hz" --steps 600 --output_dir "${OUT_DIR}"

echo "=== 4/4: E3 hand-8 (10 Hz, hold=6, seed 2) ==="
"${ISAAC_VENV}/bin/python" "${SLURM_SUBMIT_DIR}/record_task3_unified_demo.py" \
    --headless --student_run_dir "${RUNS}/student-unified-hidden-hand-8-hold6-seed2-job1160744_14/student-unified-hidden-hand-8-hold6-seed2" \
    --name "task3_e3_hand8_10hz" --steps 600 --output_dir "${OUT_DIR}"

echo "task3_demo_t4_status=PASS"
