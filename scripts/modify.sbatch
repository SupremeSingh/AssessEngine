#!/bin/bash
#SBATCH --job-name=assess-modify
#SBATCH --partition=compsci
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
export PYTHONUNBUFFERED=1
if [[ -f .env.gemini ]]; then
    source .env.gemini
fi
if [[ ! -x .venv/bin/python ]]; then
    echo 'Activate/setup the existing project Python environment first.' >&2
    exit 1
fi
.venv/bin/python -m assessengine.exploratory prepare --out "${EDIT_DIR:-runs/gemini-exploratory}"
.venv/bin/python -m assessengine.exploratory generate --out "${EDIT_DIR:-runs/gemini-exploratory}"
