#!/usr/bin/env bash
#SBATCH --job-name=shadow-demo-videos
#SBATCH --partition=gpu,gpu-long
#SBATCH --gpus=1
#SBATCH --constraint=xgpf
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Final-demo clips: runs 21_record_videos.sbatch sequentially on one GPU (one queue slot).
# DEMO_VIDEO_LIST: file with lines "<gym task> <run dir>"; line i is filmed into videos/<run>-job<JOB>_0/ (run names differ; per-line log logs/demo-video-<JOB>_<i>.log).
# The recorder plays each policy with the intent hold it was trained under (fix 2026-11-12).
# Optional third column: intent delay in control steps (evaluation-only, passed as VIDEO_INTENT_DELAY).
set -uo pipefail
: "${DEMO_VIDEO_LIST:?Submit with DEMO_VIDEO_LIST}"
mapfile -t lines < "${DEMO_VIDEO_LIST}"
fail=0
for i in "${!lines[@]}"; do
    read -r task run_dir delay <<< "${lines[$i]}"
    echo "demo_video_start index=${i} task=${task} run_dir=${run_dir} delay=${delay:-trained}"
    if SLURM_ARRAY_JOB_ID="${SLURM_JOB_ID}" SLURM_ARRAY_TASK_ID=0 VIDEO_TASK="${task}" VIDEO_RUN_DIRS="${run_dir}" VIDEO_INTENT_DELAY="${delay:-}" \
        bash "${SLURM_SUBMIT_DIR}/21_record_videos.sbatch" > "logs/demo-video-${SLURM_JOB_ID}_${i}.log" 2>&1; then
        echo "demo_video index=${i} status=PASS"
    else
        echo "demo_video index=${i} status=FAIL (logs/demo-video-${SLURM_JOB_ID}_${i}.log)"; fail=1
    fi
done
[[ ${fail} == 0 ]] && echo "demo_videos_status=PASS" || echo "demo_videos_status=FAIL"
exit ${fail}
