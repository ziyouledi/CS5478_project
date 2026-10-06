#!/usr/bin/env bash
#SBATCH --job-name=isaac-probe
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=xgpf
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -Eeuo pipefail

echo "record_type=cluster_probe"
echo "timestamp=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname -f)"
echo "partition=${SLURM_JOB_PARTITION}"
echo "cpus=${SLURM_CPUS_PER_TASK}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "slurm_job_gpus=${SLURM_JOB_GPUS:-unset}"
echo "slurm_step_gpus=${SLURM_STEP_GPUS:-unset}"

echo "--- os ---"
uname -a
cat /etc/os-release
ldd --version | head -n 1

echo "--- gpu-summary ---"
gpu_status=0
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader || gpu_status=$?

echo "--- gpu-full ---"
nvidia-smi || gpu_status=$?

echo "--- gpu-device-files ---"
ls -l /dev/nvidia* 2>&1 || true

echo "--- slurm-allocation ---"
scontrol show job "${SLURM_JOB_ID}" -dd 2>&1 || true
scontrol show node "$(hostname -s)" 2>&1 || true

echo "--- tools ---"
for tool in python3 python3.11 conda mamba micromamba uv apptainer singularity docker git cmake gcc; do
    if command -v "${tool}" >/dev/null 2>&1; then
        printf '%s=%s\n' "${tool}" "$(command -v "${tool}")"
        "${tool}" --version 2>&1 | head -n 1 || true
    else
        printf '%s=missing\n' "${tool}"
    fi
done

echo "--- storage ---"
df -h "${HOME}"
quota -s 2>&1 || true

echo "--- outbound-https ---"
if command -v curl >/dev/null 2>&1; then
    curl -LIsS --max-time 15 https://pypi.org/simple/ | head -n 1 || echo "pypi_org_https=failed"
    curl -LIsS --max-time 15 https://pypi.nvidia.com/ | head -n 1 || echo "pypi_nvidia_https=failed"
else
    echo "curl=missing"
fi

if [[ "${gpu_status}" -eq 0 ]]; then
    echo "probe_status=PASS"
else
    echo "probe_status=FAIL_NO_VISIBLE_GPU"
    exit "${gpu_status}"
fi
