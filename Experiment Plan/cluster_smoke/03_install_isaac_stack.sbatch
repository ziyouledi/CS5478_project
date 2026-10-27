#!/usr/bin/env bash
#SBATCH --job-name=isaac-install
#SBATCH --partition=normal
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -Eeuo pipefail

STACK="${STACK:-${HOME}/software/isaaclab-stack}"
VENV_DIR="${STACK}/env_isaaclab"
ISAACLAB_DIR="${STACK}/IsaacLab"
UV="${STACK}/bin/uv"

[[ -x "${UV}" ]] || { echo "missing_uv=${UV}"; exit 1; }
[[ -x "${VENV_DIR}/bin/python" ]] || { echo "missing_venv=${VENV_DIR}"; exit 1; }

export PATH="${STACK}/bin:${VENV_DIR}/bin:${PATH}"
export UV_PYTHON_INSTALL_DIR="${STACK}/python"
export UV_CACHE_DIR="/tmp/${USER}/uv-cache-${SLURM_JOB_ID}"
export UV_LINK_MODE=copy
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
mkdir -p "${UV_CACHE_DIR}"

echo "record_type=isaac_stack_install"
echo "timestamp=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname -f)"
echo "stack=${STACK}"
echo "uv=$(${UV} --version)"
echo "python=$(${VENV_DIR}/bin/python --version 2>&1)"
echo "uv_cache=${UV_CACHE_DIR}"
echo "uv_link_mode=${UV_LINK_MODE}"

source "${VENV_DIR}/bin/activate"

echo "--- install-isaac-sim ---"
"${UV}" pip install --python "${VENV_DIR}/bin/python" \
  "isaacsim[all,extscache]==5.1.0" \
  --extra-index-url https://pypi.nvidia.com

echo "--- install-pytorch ---"
"${UV}" pip install --python "${VENV_DIR}/bin/python" --upgrade \
  torch==2.7.0 \
  torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128

echo "--- clone-isaac-lab ---"
if [[ ! -d "${ISAACLAB_DIR}/.git" ]]; then
  git clone --branch v2.3.1 --depth 1 \
    https://github.com/isaac-sim/IsaacLab.git "${ISAACLAB_DIR}"
else
  current_tag=$(git -C "${ISAACLAB_DIR}" describe --tags --exact-match 2>/dev/null || true)
  if [[ "${current_tag}" != "v2.3.1" ]]; then
    echo "existing_isaaclab_checkout_has_unexpected_tag=${current_tag:-none}"
    exit 1
  fi
fi

echo "--- install-isaac-lab-rl-games ---"
cd "${ISAACLAB_DIR}"
./isaaclab.sh --install rl_games

echo "--- installed-versions ---"
python - <<'PY'
import sys
import torch

print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"torch_cuda_build={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
PY
python -m pip freeze > "${STACK}/logs/pip-freeze.txt"
git rev-parse HEAD | tee "${STACK}/logs/isaaclab-commit.txt"

echo "install_status=PASS"
