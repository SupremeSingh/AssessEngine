#!/bin/bash
#SBATCH --job-name=assess-baseline
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PWD/.cache/pip}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# Set PYTHON_BIN to a Python 3.10/3.11/3.12 executable if needed.
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "Use Python 3.10-3.12"'
nvidia-smi
# Fail before downloading dependencies/weights if scheduled on an older small GPU.
"$PYTHON_BIN" - <<'PYGPU'
import subprocess
memory = int(subprocess.check_output([
    'nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'
], text=True).splitlines()[0])
if memory < 20000:
    raise SystemExit('Request an A5000 (24 GB), A6000 (48 GB), or V100 (32 GB). '
                     'Use sinfo to find the corresponding GPU resource name.')
PYGPU
if [[ ! -d .venv ]]; then
    "$PYTHON_BIN" -m venv .venv
fi
source .venv/bin/activate
# Dependency installation and model inference happen on the compute node.
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
mkdir -p "${RUN_DIR:-runs/baseline}"
python -m pip freeze > "${RUN_DIR:-runs/baseline}/environment.txt"
python -m assessengine prepare --questions "${QUESTIONS:-10}" --data "${DATA_DIR:-data/mohler}"
python -m assessengine generate --data "${DATA_DIR:-data/mohler}" \
    --run "${RUN_DIR:-runs/baseline}" --samples "${SAMPLES:-3}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-2048}" --precision "${PRECISION:-4bit}"
# Separate processes release DeepSeek GPU memory before loading Desklib.
python -m assessengine score --run "${RUN_DIR:-runs/baseline}"
python -m assessengine report --run "${RUN_DIR:-runs/baseline}"
