#!/bin/bash
#SBATCH --job-name=assess-rewrites
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
DATA_DIR="${DATA_DIR:-data/sas-exploratory-v1}"
RUN_DIR="${RUN_DIR:-runs/sas-exploratory-221-v1}"

# Reuse installed dependencies and cached model. No Gemini calls or dataset downloads.
.venv/bin/python -m assessengine.exploratory materialize --data "$DATA_DIR"
mkdir -p "$RUN_DIR"
.venv/bin/python -m pip freeze > "$RUN_DIR/environment.txt"
.venv/bin/python -m assessengine generate --data "$DATA_DIR" --run "$RUN_DIR" \
    --samples 50 --samples-by-set experiments/inputs/rewrite-quotas-v1.json \
    --resume-from runs/sas-exploratory-v1 \
    --max-attempts 5 --max-new-tokens 4096 --precision 4bit \
    --revision 1df8507178afcc1bef68cd8c393f61a886323761
.venv/bin/python -m assessengine.snapjudge export --run "$RUN_DIR"
