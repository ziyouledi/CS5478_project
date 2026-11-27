#!/usr/bin/env bash
#SBATCH --job-name=isaac-finalize
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -Eeuo pipefail

STACK="${STACK:-${HOME}/software/isaaclab-stack}"
VENV_DIR="${STACK}/env_isaaclab"
ISAACLAB_DIR="${STACK}/IsaacLab"
UV="${STACK}/bin/uv"

[[ -x "${UV}" ]] || { echo "missing_uv=${UV}"; exit 1; }
[[ -x "${VENV_DIR}/bin/python" ]] || { echo "missing_venv=${VENV_DIR}"; exit 1; }
[[ -d "${ISAACLAB_DIR}/.git" ]] || { echo "missing_isaaclab_repo=${ISAACLAB_DIR}"; exit 1; }

export PATH="${STACK}/bin:${VENV_DIR}/bin:${PATH}"
export UV_PYTHON_INSTALL_DIR="${STACK}/python"
export UV_CACHE_DIR="/tmp/${USER}/uv-cache-${SLURM_JOB_ID}"
export UV_LINK_MODE=copy
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
mkdir -p "${UV_CACHE_DIR}" "${STACK}/logs"
source "${VENV_DIR}/bin/activate"

echo "record_type=isaac_stack_finalize"
echo "timestamp=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname -f)"

echo "--- restore-final-compatibility-pins ---"
"${UV}" pip install --python "${VENV_DIR}/bin/python" \
  click==8.1.7 \
  filelock==3.13.1 \
  fsspec==2024.6.1 \
  idna==3.10 \
  ipython==8.37.0 \
  markupsafe==2.1.3 \
  networkx==3.3 \
  numpy==1.26.0 \
  pillow==11.3.0 \
  psutil==5.9.8 \
  starlette==0.49.1 \
  sympy==1.13.3 \
  typing-extensions==4.12.2

echo "--- verify-package-metadata ---"
python - <<'PY'
import importlib.metadata as md
import importlib.util
import sys
import torch

print(f"python={sys.version.split()[0]}")
for distribution in ("isaacsim", "isaaclab", "isaaclab_tasks", "isaaclab_rl", "rl-games"):
    print(f"{distribution}={md.version(distribution)}")
print(f"torch={torch.__version__}")
print(f"torch_cuda_build={torch.version.cuda}")
print(f"cuda_available_on_cpu_node={torch.cuda.is_available()}")

for module in ("isaaclab", "isaaclab_tasks", "isaaclab_rl"):
    spec = importlib.util.find_spec(module)
    if spec is None:
        raise ModuleNotFoundError(module)
    print(f"module_location[{module}]={spec.origin}")
PY

echo "--- pip-check-with-known-conflicts ---"
python - <<'PY'
import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-m", "pip", "check"],
    check=False,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)
lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
for line in lines:
    print(line)

allowed_prefixes = (
    "fastapi 0.115.7 has requirement starlette<0.46.0,>=0.40.0, but you have starlette 0.49.1.",
    "wandb 0.30.0 has requirement click>=8.2.0, but you have click 8.1.7.",
    "huggingface-hub 1.33.0 has requirement click<9.0.0,>=8.4.2, but you have click 8.1.7.",
)
unexpected = [line for line in lines if line not in allowed_prefixes and line != "No broken requirements found."]
if unexpected:
    print("unexpected_dependency_conflicts=FAIL")
    raise SystemExit(1)
print(f"known_metadata_conflicts={sum(line in allowed_prefixes for line in lines)}")
print("dependency_gate=PASS")
PY
python -m pip freeze > "${STACK}/logs/pip-freeze.txt"
git -C "${ISAACLAB_DIR}" rev-parse HEAD | tee "${STACK}/logs/isaaclab-commit.txt"

echo "finalize_status=PASS"
