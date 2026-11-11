#!/bin/bash
#SBATCH --job-name=assess-score
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
# Run on CPU without loading DeepSeek or repeating answer generation.
export CUDA_VISIBLE_DEVICES=""
if [[ ! -x .venv/bin/python ]]; then
    echo 'Missing .venv; install requirements in a compute allocation first.' >&2
    exit 1
fi
.venv/bin/python -m assessengine.snapjudge run --run "${RUN_DIR:-runs/sas-baseline}" \
    --key-file "${SNAPJUDGE_KEY_FILE:-$HOME/.config/assessengine/snapjudge.key}"
