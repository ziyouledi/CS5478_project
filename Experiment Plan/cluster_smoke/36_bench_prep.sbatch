#!/usr/bin/env bash
#SBATCH --job-name=bench-prep
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Data for bench_vlm_eval.py (Z1 pipeline check): pyarrow into the vLLM env, and public benchmark parquet files
# into ~/data/low-dim-intent/hf/datasets/{blink,mmstar,embspatial}.

set -Eeuo pipefail
VENV_DIR="${HOME}/software/vlm-stack/env_vllm"
UV="${HOME}/software/isaaclab-stack/bin/uv"
DATA_DIR="${HOME}/data/low-dim-intent/hf/datasets"
export UV_CACHE_DIR="/tmp/${USER}/uv-cache-${SLURM_JOB_ID}" UV_LINK_MODE=copy
export HF_HOME="${HOME}/data/low-dim-intent/hf/cache" HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1
mkdir -p "${DATA_DIR}" "${UV_CACHE_DIR}"
echo "record_type=bench_prep job_id=${SLURM_JOB_ID} node=$(hostname -f) timestamp=$(date --iso-8601=seconds)"

"${UV}" pip install --python "${VENV_DIR}/bin/python" pyarrow
"${VENV_DIR}/bin/python" - "${DATA_DIR}" <<'PY'
import sys
from huggingface_hub import snapshot_download
root = sys.argv[1]
for name, repo, patterns in [
    ("blink", "BLINK-Benchmark/BLINK", ["*/val-*.parquet", "README.md"]),
    ("mmstar", "Lin-Chen/MMStar", ["mmstar.parquet", "README.md"]),
    ("embspatial", "FlagEval/EmbSpatial-Bench", ["data/*.parquet", "README.md"]),
]:
    path = snapshot_download(repo_id=repo, repo_type="dataset", local_dir=f"{root}/{name}", allow_patterns=patterns)
    print(f"downloaded={name} path={path}", flush=True)
PY
du -sh "${DATA_DIR}"/*
rm -rf "${UV_CACHE_DIR}"
echo "bench_prep_status=PASS"
