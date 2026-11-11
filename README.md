# AssessEngine

A small baseline comparing authentic CS student answers with answers from
`deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`, scored with
`desklib/ai-text-detector-v1.01`. No paid APIs or training. Gemini modifications
are a later experiment, not part of this baseline.

## Cluster quick start

Push this repository locally, then on Duke:

```bash
cd /usr/xtmp/ms785/AssessEngine
git pull
python3 --version
sinfo -o '%P %G %a'
mkdir -p logs
sbatch scripts/baseline.sbatch
```

Requirements: Linux, NVIDIA GPU, CUDA-compatible driver for PyTorch 2.6 (CUDA
12.4 wheels), Python **3.10–3.12** with `venv`, internet access from the compute
node for the first install/download, and approximately **60 GB free disk**.
If `python3` is not a supported version, load a suitable Duke Python module
(`module avail`) or export `PYTHON_BIN` to an installed interpreter before submission.
Do not run inference on the login node.

**GPU selection:** the script requests one GPU on the default partition. Use
an **A5000 24 GB, A6000 48 GB, or V100 32 GB**. It deliberately stops on smaller
GPUs. If the default partition does not offer GPUs, or offers older cards, use
the partition and GPU type shown by `sinfo`:

```bash
# Replace PARTITION and TYPE with exact values from sinfo; these are placeholders.
sbatch --partition=PARTITION --gres=gpu:TYPE:1 scripts/baseline.sbatch
```

Duke's actual partition/type names and any required account cannot be inferred
from the hardware description. Command-line Slurm flags override script defaults.
No cluster login, GPU allocation, installation, or inference has been verified
from the local development machine.

The job installs the pinned dependencies in `.venv`, then prepares **10 questions**,
generates **3 responses per question**, scores humans and completed AI answers,
and writes a report. It uses NF4 4-bit loading, temperature 0.6, top-p 0.95,
and a 2,048-token cap including reasoning. Quantization happens while loading
the official checkpoint: the initial download still includes the full weights
(approximately 30 GB). Dependencies/downloads are cached for later jobs.
Queue time depends on Duke; the first job includes installation and downloads.
The two-hour request is a limit, not a runtime promise.

Monitor:

```bash
squeue -u "$USER"
# Substitute the job ID printed by sbatch.
tail -f logs/assess-baseline-JOB_ID.out
```

First run a smaller smoke test if desired:

```bash
QUESTIONS=2 SAMPLES=1 DATA_DIR=data/smoke RUN_DIR=runs/smoke sbatch scripts/baseline.sbatch
```

Then submit the default command above for the 10-question pilot. Each dataset
selection and generation configuration needs its own data/run directories.
Use just one writer per run directory; do not submit overlapping jobs to it.

## Outputs

- `data/mohler/dataset.json`: dataset revision and selection configuration.
- `data/mohler/questions.jsonl`: exact questions, raw student responses, grades, IDs.
- `runs/baseline/questions.jsonl`: copy of the selected inputs.
- `runs/baseline/generation.json`: model revision, input hash, generation settings, environment.
- `runs/baseline/answers.jsonl`: rendered prompts, raw outputs, reasoning, final answers,
  completion status, seeds, GPU, tokens, and elapsed generation time.
- `runs/baseline/detector.json`: detector revision and input hashes.
- `runs/baseline/scores.jsonl` and `scores.csv`: per-answer scores, authorship,
  word/token counts, grades, and detector truncation flags.
- `runs/baseline/report.json`: exploratory AUROC, false-positive fraction for humans,
  detection fraction for AI at threshold 0.5, lengths, and excluded generations.
- `runs/baseline/environment.txt`: installed package versions.

Completed generations are appended and flushed immediately. Resubmit the same
command to resume after a normal interruption. An incompatible configuration is
rejected rather than mixed into an existing run. Do not edit JSONL files while
jobs are running. A partially written last JSONL line after a hard filesystem
failure produces an explicit parse error rather than silent data loss.
Outputs, caches, and environments are gitignored. **Back up results separately:**
the cluster's xtmp storage is not backed up.

## Recover from a detector download failure

If generation finished but Desklib failed with HTTP 429, keep all run files.
Hugging Face may rate-limit a shared cluster IP; authenticate using a free
account and a read token from https://huggingface.co/settings/tokens.
Run login interactively on the cluster, using the same cache location as the jobs:

```bash
cd /usr/xtmp/ms785/AssessEngine
module load python/3.12.8
source .venv/bin/activate
export HF_HOME="$PWD/.cache/huggingface"
huggingface-cli login
```

Paste the token only into the CLI's hidden prompt, never into source files,
chat, or shell command arguments. Choose **N** when asked to add it as a Git
credential. The login is persisted under HF_HOME and automatically used by
Hugging Face downloads. No paid subscription is needed. An existing HF_TOKEN
environment variable takes precedence over the saved login.

Then finish scoring on a CPU compute node, without repeating generation:

```bash
mkdir -p logs
sbatch scripts/score.sbatch
```

Follow `logs/assess-score-JOB_ID.out`, then inspect
`runs/baseline/report.json`. No regeneration or manifest deletion is needed.
The scoring job uses the already installed environment; it does not reinstall
packages. CPU scoring is slower than GPU scoring but requires no GPU allocation.
Authentication does not guarantee immediate removal of a rate limit; if 429
persists, wait for the limit to reset before resubmitting this scoring-only job.

## Separate stages

Inside an appropriate compute allocation, after installing the requirements:

```bash
source .venv/bin/activate
export HF_HOME="$PWD/.cache/huggingface"
python -m assessengine prepare --questions 10
python -m assessengine generate --samples 3
python -m assessengine score
python -m assessengine report
```

Generation requires CUDA; scoring can also run on CPU. Each stage exposes
`--help`. Model/dataset revisions resolve to commit hashes and are reused on
resume. You can explicitly supply `--revision COMMIT` to prepare/generate/score.
Per-response seeds support resumption; exact bitwise reproduction across GPU
architectures/library versions is not guaranteed.

For 16-bit generation on a 48 GB A6000, use a new run:

```bash
PRECISION=16bit RUN_DIR=runs/baseline-16bit sbatch scripts/baseline.sbatch
```

If many generations reach the cap, inspect their status and use a **new run**
with a larger limit, for example:

```bash
MAX_NEW_TOKENS=4096 RUN_DIR=runs/baseline-4096 sbatch scripts/baseline.sbatch
```

## Method and limitations

- Uses the dataset's **raw/open_ended** subset; student text is not corrected or
  rewritten. Selection is deterministic with seed 42. Identical question wordings
  are grouped, retaining original source IDs and responses. No train/test split
  is claimed at this stage; reserve question families before optimization.
- DeepSeek receives the **exact question only**, through its official chat template.
  It receives no student answers, reference answers, or system instructions.
  Some questions may need missing diagrams/programs/context; inspect selected
  questions before interpreting results. There is no automatic validity claim.
- Reasoning is stored separately. Only outputs with a closing `</think>` boundary,
  a nonempty final answer, and an EOS are scored. Capped and malformed outputs
  are counted and excluded, without replacing failed samples selectively.
- Desklib uses its documented masked mean-pooling/single-logit architecture,
  implemented locally; checkpoint key mismatches fail. Remote repository code
  is not executed. Scores are detector outputs, not verified authorship probabilities.
- Questions are short and human/AI answer lengths and correctness may differ.
  Report word counts and review these confounds. No human text is padded to meet
  detector lengths. Desklib truncation is explicit at 768 tokens.
- This is a pooled descriptive baseline with unequal class counts and correlated
  answers within questions, not a statistical significance test or proof of
  educational validity. Public questions may have appeared in model training.
- Prompt optimization and matched-human validation are intentionally future stages.
  Original human answers cannot stand in for humans answering modified questions.

## Sources and attribution

- [MohlerASAG dataset card](https://huggingface.co/datasets/nkazi/MohlerASAG)
  (card lists CC BY 4.0). Cite **Michael Mohler, Razvan Bunescu, and Rada Mihalcea
  (2011), Learning to Grade Short Answer Questions using Semantic Similarity
  Measures and Dependency Graph Alignments**, [ACL paper](https://aclanthology.org/P11-1076/),
  and credit the Hugging Face conversion by Nazmul Kazi. Dataset content is kept
  outside Git; consult original source terms before redistributing it.
- [DeepSeek model card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)
  (MIT; distilled from Qwen, not the full DeepSeek-R1).
- [Desklib model card](https://huggingface.co/desklib/ai-text-detector-v1.01)
  (MIT; architecture and scoring procedure).

## Local checks (no model downloads)

```bash
python3 -m unittest discover -s tests -v
bash -n scripts/baseline.sbatch
```
