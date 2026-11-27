#!/usr/bin/env bash
#SBATCH --job-name=vllm-srv-c1
#SBATCH --partition=gpu,gpu-long
#SBATCH --gpus=h100-47:1
#SBATCH --exclude=xgpi15,xgpj0
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Serves a merged HF VLM over cluster-internal TCP for the T4 Isaac Lab closed-loop client (46_isaac_closedloop_t4.sbatch).
# Writes http://<host_ip>:<port> to ENDPOINT_FILE once ready, and exits cleanly when DONE_FLAG appears.

set -Eeuo pipefail

VLLM_VENV="${HOME}/software/vlm-stack/env_vllm"
HF_DIR="${HOME}/data/low-dim-intent/hf"
MODEL_DIR="${CL_MODEL_DIR:-${HOME}/data/low-dim-intent/runs/c1-sft-vision-full-job1173965/merged_hf}"
COORD_DIR="${CL_COORD_DIR:-${HOME}/data/low-dim-intent/runs/isaac-cl-coord}"
PORT=$(( 18200 + (SLURM_JOB_ID % 500) ))

mkdir -p "${COORD_DIR}" logs
ENDPOINT_FILE="${COORD_DIR}/endpoint.txt"
DONE_FLAG="${COORD_DIR}/done.flag"
rm -f "${ENDPOINT_FILE}" "${DONE_FLAG}"

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_DIR}/cache"
export VLLM_CACHE_ROOT="${HOME}/data/low-dim-intent/cache/vllm"

NODE_IP="$(hostname -I | awk '{print $1}')"
echo "record_type=vllm_srv job_id=${SLURM_JOB_ID} node=$(hostname -f) ip=${NODE_IP} port=${PORT} model=${MODEL_DIR}"

VLLM_PID=""
cleanup() {
    rm -f "${ENDPOINT_FILE}"
    if [[ -n "${VLLM_PID}" ]]; then
        kill "${VLLM_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

"${VLLM_VENV}/bin/vllm" serve "${MODEL_DIR}" \
    --served-model-name "vlm_planner" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --gpu-memory-utilization 0.75 \
    --max-model-len 8192 \
    --limit-mm-per-prompt '{"image": 1}' \
    --enforce-eager \
    --disable-log-requests \
    > "${COORD_DIR}/vllm_server_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!

for i in $(seq 1 300); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "vLLM server ready after $((i * 2))s at http://${NODE_IP}:${PORT}"
        echo "http://${NODE_IP}:${PORT}" > "${ENDPOINT_FILE}"
        break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "ERROR: vLLM server exited prematurely!"
        tail -n 40 "${COORD_DIR}/vllm_server_${SLURM_JOB_ID}.log"
        exit 1
    fi
    sleep 2
done
test -f "${ENDPOINT_FILE}"

echo "Waiting for client done flag at ${DONE_FLAG}..."
while [[ ! -f "${DONE_FLAG}" ]]; do
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "ERROR: vLLM server died while waiting for client!"
        tail -n 40 "${COORD_DIR}/vllm_server_${SLURM_JOB_ID}.log"
        exit 1
    fi
    sleep 10
done

echo "Client finished (done flag seen). Shutting down vLLM server cleanly."
echo "vllm_srv_status=PASS"
