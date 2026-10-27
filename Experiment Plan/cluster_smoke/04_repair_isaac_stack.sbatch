#!/usr/bin/env bash
#SBATCH --job-name=isaac-repair
#SBATCH --partition=normal
#SBATCH --time=01:30:00
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
[[ -d "${ISAACLAB_DIR}/.git" ]] || { echo "missing_isaaclab_repo=${ISAACLAB_DIR}"; exit 1; }

export PATH="${STACK}/bin:${VENV_DIR}/bin:${PATH}"
export UV_PYTHON_INSTALL_DIR="${STACK}/python"
export UV_CACHE_DIR="/tmp/${USER}/uv-cache-${SLURM_JOB_ID}"
export UV_LINK_MODE=copy
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
mkdir -p "${UV_CACHE_DIR}" "${STACK}/logs"
source "${VENV_DIR}/bin/activate"

echo "record_type=isaac_stack_repair"
echo "timestamp=$(date --iso-8601=seconds)"
echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname -f)"
echo "isaaclab_commit=$(git -C "${ISAACLAB_DIR}" rev-parse HEAD)"

echo "--- apply-upstream-flatdict-fix ---"
ISAACLAB_SETUP="${ISAACLAB_DIR}/source/isaaclab/setup.py"
python - "${ISAACLAB_SETUP}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
if '"flatdict==4.0.1"' in text:
    path.write_text(text.replace('"flatdict==4.0.1"', '"flatdict==4.0.0"'))
elif '"flatdict==4.0.0"' not in text:
    raise SystemExit("expected flatdict pin not found in Isaac Lab setup.py")
PY
git -C "${ISAACLAB_DIR}" diff -- source/isaaclab/setup.py | tee "${STACK}/logs/flatdict-v2.3.1.patch"

echo "--- establish-compatible-build-dependencies ---"
"${UV}" pip install --python "${VENV_DIR}/bin/python" \
  flatdict==4.0.0 \
  ipython==8.37.0 \
  click==8.1.7 \
  numpy==1.26.0 \
  pillow==11.3.0 \
  psutil==5.9.8 \
  typing-extensions==4.12.2

echo "--- resume-isaac-lab-rl-games-install ---"
cd "${ISAACLAB_DIR}"
./isaaclab.sh --install rl_games

echo "--- restore-isaac-sim-compatible-pins ---"
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

echo "--- verify-imports-and-versions ---"
python - <<'PY'
import importlib.metadata as md
import importlib.util
import sys
import torch

print(f"python={sys.version.split()[0]}")
print(f"isaacsim={md.version('isaacsim')}")
print(f"isaaclab={md.version('isaaclab')}")
print(f"isaaclab_tasks={md.version('isaaclab_tasks')}")
print(f"rl_games={md.version('rl-games')}")
print(f"torch={torch.__version__}")
print(f"torch_cuda_build={torch.version.cuda}")
print(f"cuda_available_on_cpu_node={torch.cuda.is_available()}")
for module in ("isaaclab", "isaaclab_tasks", "isaaclab_rl"):
    spec = importlib.util.find_spec(module)
    if spec is None:
        raise ModuleNotFoundError(module)
    print(f"module_location[{module}]={spec.origin}")
PY

echo "--- pip-check-report ---"
python -m pip check || true
python -m pip freeze > "${STACK}/logs/pip-freeze.txt"
git rev-parse HEAD | tee "${STACK}/logs/isaaclab-commit.txt"

echo "repair_status=PASS"
