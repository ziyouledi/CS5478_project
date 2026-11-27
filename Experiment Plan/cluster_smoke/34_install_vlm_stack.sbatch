#!/usr/bin/env bash
#SBATCH --job-name=vlm-install
#SBATCH --partition=normal
#SBATCH --time=02:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# vLLM environment and model weights for Z1 (GOAL_LEVEL_RL_PLAN.md). Separate from the Isaac stack.
# Env: ~/software/vlm-stack/env_vllm (Python 3.12, vllm pinned). Weights: ~/data/low-dim-intent/hf/<repo name>.
# VLM_MODELS overrides the repositories (space-separated).

set -Eeuo pipefail

STACK="${HOME}/software/vlm-stack"
VENV_DIR="${STACK}/env_vllm"
UV="${HOME}/software/isaaclab-stack/bin/uv"
HF_DIR="${HOME}/data/low-dim-intent/hf"
VLLM_VERSION="${VLLM_VERSION:-0.11.0}"
MODELS="${VLM_MODELS:-Qwen/Qwen3-VL-8B-Thinking Qwen/Qwen3-VL-8B-Instruct Qwen/Qwen2.5-VL-7B-Instruct}"

export UV_PYTHON_INSTALL_DIR="${STACK}/python"
export UV_CACHE_DIR="/tmp/${USER}/uv-cache-${SLURM_JOB_ID}"
export UV_LINK_MODE=copy
export HF_HOME="${HF_DIR}/cache"
export PYTHONUNBUFFERED=1
mkdir -p "${STACK}" "${HF_DIR}" "${UV_CACHE_DIR}"

echo "record_type=vlm_stack_install"
echo "timestamp=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname -f)"
echo "vllm_version=${VLLM_VERSION}"
echo "models=${MODELS}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    "${UV}" python install 3.12
    "${UV}" venv --python 3.12 "${VENV_DIR}"
fi
# transformers 5.x breaks vllm 0.11.0 (Qwen2Tokenizer.all_special_tokens_extended missing; rope_type/mrope config
# conflict for Qwen2.5-VL; job 1164383), so pin the 4.57 line, the first with Qwen3-VL.
"${UV}" pip install --python "${VENV_DIR}/bin/python" "vllm==${VLLM_VERSION}" "transformers>=4.57.1,<4.58" \
    "huggingface_hub[hf_transfer]<1.0" pillow
"${VENV_DIR}/bin/python" - <<'EOF'
import importlib.metadata as md
for name in ("vllm", "torch", "transformers", "huggingface_hub"):
    print(f"pkg_{name}={md.version(name)}")
EOF

export HF_HUB_ENABLE_HF_TRANSFER=1
[[ "${SKIP_DOWNLOAD:-0}" == 1 ]] && MODELS=""
for repo in ${MODELS}; do
    target="${HF_DIR}/${repo#*/}"
    echo "--- download ${repo} -> ${target} ---"
    "${VENV_DIR}/bin/python" - "${repo}" "${target}" <<'EOF'
import sys
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2],
                         allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.py", "*.tiktoken"])
print(f"downloaded={path}")
EOF
    du -sh "${target}"
done
rm -rf "${UV_CACHE_DIR}"
echo "vlm_install_status=PASS"
