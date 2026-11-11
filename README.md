# AssessEngine

A reproducible **ASAP-SAS science/biology pilot**: authentic student answers,
DeepSeek-R1-Distill-Qwen-14B, and SnapJudge via its official batch client.
No paid model API is used. Gemini optimization is a later stage.

## What this version runs

- **Five prompts:** SAS essay sets **1, 2, 5, 6, 10** (science and biology).
- **250 human answers:** 50 per prompt, sampled with seed 42 from exact-text
  deduplicated responses containing at least 50 whitespace-separated words.
- **50 AI attempts:** 10 per prompt, using 4-bit DeepSeek 14B on one A5000.
- **50–100 word final-answer instruction** for every AI prompt. This requests
  comparable length; it does not enforce length or change human answers.
- **4,096 generation tokens**, including reasoning. Failed, capped, and short
  answers are retained and excluded transparently; they are not selectively replaced.
- **SnapJudge** runs separately, with `--no-save`, three simultaneous submissions,
  no GPU, and your private SnapJudge key file.

Verified locally: real pinned dataset preparation produced all five prompts and
250 eligible human responses; tests cover rubric removal, context preservation,
length filtering, deterministic sampling, duplicate removal, and score association.
GPU generation on SAS and live SnapJudge scoring must still run on the cluster.
Old Mohler data and results remain untouched in their old directories.

## Run on Duke

Push these local changes, then on the cluster:

```bash
cd /usr/xtmp/ms785/AssessEngine
git pull
module load python/3.12.8
export PYTHON_BIN="$(command -v python3)"
export HF_HOME="$PWD/.cache/huggingface"
mkdir -p logs
sbatch scripts/baseline.sbatch
```

The script now explicitly requests `compsci-gpu`, one `a5000`, four CPU cores,
48 GB host memory, and a **four-hour limit**. It creates/reuses `.venv`, installs
pinned requirements, prepares the dataset, generates responses, and exports the
SnapJudge input files. It **does not submit anything to SnapJudge**.
Use `sbatch --gres=gpu:a6000:1 scripts/baseline.sbatch` to request an A6000 instead.
Queue time depends on Duke. First download needs internet access; approximately
60 GB disk accommodates full checkpoint weights and dependencies. 4-bit loading
still downloads the original full checkpoint. Python 3.10–3.12 and a compatible
NVIDIA driver for the pinned PyTorch CUDA wheels are required.

Follow the job (replace JOB_ID):

```bash
squeue -j JOB_ID
tail -f logs/assess-baseline-JOB_ID.out
```

After successful generation, with your SnapJudge key already saved:

```bash
sbatch scripts/score.sbatch
```

Or submit the pair together, making scoring depend on successful generation:

```bash
GEN_JOB=$(sbatch --parsable scripts/baseline.sbatch)
sbatch --dependency=afterok:"$GEN_JOB" scripts/score.sbatch
```

Do not submit both the standalone generation command and this pair: choose one
workflow. Do not run overlapping jobs writing the same output directory.
The scoring job uses CPU partition `compsci` and the existing `.venv`.
It loads no detector weights locally; detection is through the remote service.

Then read **the new SAS report**:

```bash
cat runs/sas-baseline/snapjudge/report.json
```

The old `runs/baseline/report.json` is the historical Mohler/Desklib report.

## SnapJudge key (once)

Get your separately issued **sjk_** key from the SnapJudge team. Hugging Face
credentials do not work. The guide does not document self-service key issuance.
Save it privately on the cluster; never paste it in chat or command arguments:

```bash
source .venv/bin/activate
python - <<'PY'
from pathlib import Path
import getpass
p = Path.home() / '.config/assessengine/snapjudge.key'
p.parent.mkdir(parents=True, exist_ok=True)
key = getpass.getpass('SnapJudge key: ').strip()
if not key.startswith('sjk_'):
    raise SystemExit('Expected a SnapJudge sjk_ key')
p.touch(mode=0o600, exist_ok=True)
p.chmod(0o600)
p.write_text(key + '\n')
PY
```

`score.sbatch` passes the **file path**, not its contents. Override with the
`SNAPJUDGE_KEY_FILE` environment variable if needed. The official batch client
is downloaded once and cached; its URL and SHA-256 are logged. The service's
`--no-save` option requests analysis without retaining the submissions.
If Hugging Face rate-limits checkpoint/dataset downloads, log in via
`huggingface-cli login` after activating `.venv` and setting HF_HOME as above.
Do not add a Git credential helper just for model downloads.

## Dataset preparation and prompt integrity

Source: `nlpatunt/D_ASAP-SAS`, pinned to commit
`f97ad6415c94c763b901423b72f78f1fbf46d630`. It repackages the authentic,
pre-ChatGPT ASAP-SAS student-response data; score labels are removed by the
mirror. We do not synthesize, correct, pad, or concatenate student answers.

The mirror's **prompt fields include rubrics in sets 5 and 6**. Preparation
removes the explicit rubric section and the set-5 item title. Sets 1, 2, and 10
retain their experiment procedures and data tables from the mirror's textual
transcription. The original field hash, source URL, and optional image URL are
logged. Set 10 includes an image URL; generation uses the mirror's textual
representation, not a vision model. We inspected the text, but have not
independently validated it against every original assessment image. No rubric,
human response, grading label, or reference answer is given to DeepSeek.
Changed or inconsistent rubric boundaries fail preparation.

| Set | Topic | Unique eligible responses available | Selected |
|---|---|---:|---:|
| 1 | Replicating an experiment | 696 | 50 |
| 2 | Plastic stretchability investigation | 849 | 50 |
| 5 | Protein synthesis | 194 | 50 |
| 6 | Cell-membrane transport | 206 | 50 |
| 10 | Lid color, temperature, and doghouse paint | 506 | 50 |

We pool the mirror's train/validation/test partitions for this descriptive pilot;
original split and response IDs are retained. These are **answer-level splits**,
not held-out questions. Before Gemini optimization, reserve prompt families for
independent evaluation. With only five science prompts, broad transfer claims
are not supported. This is science education, not CS education.

Selecting only long human answers changes the population. The added length
instruction applies to AI answers only and is a documented generation condition,
not evidence that it matches the original student setting. Compare actual lengths
and correctness before attributing score differences to authorship.

## Outputs and resuming

- `data/sas/dataset.json`: pinned revision and sampling configuration.
- `data/sas/coverage.json`: pool, eligibility, deduplication and sampling counts.
- `data/sas/questions.jsonl`: cleaned prompts and raw sampled human answers.
- `runs/sas-baseline/generation.json`: settings, revision, input hash and environment.
- `runs/sas-baseline/answers.jsonl`: rendered prompts, raw output, reasoning,
  final answers, completion status, per-response seeds, GPU, tokens and timing.
- `runs/sas-baseline/snapjudge/coverage.json`: actual submission eligibility.
- `runs/sas-baseline/snapjudge/index.jsonl`: opaque file names mapped to answer IDs.
- `runs/sas-baseline/snapjudge/results.csv`, `results_json/`: official raw results.
- `runs/sas-baseline/snapjudge/scores.jsonl`: results joined to local authorship.
- `runs/sas-baseline/snapjudge/report.json`: scores, verdicts, coverage and exclusions.

Generation flushes each attempt to disk. Re-submit the same command to resume;
completed attempts are not regenerated. Changed settings are rejected; use new
run/data directories for new configurations. SnapJudge's client also resumes
finished documents. Do not edit exported files or inputs under a running job.
All attempts are preserved; short or incomplete outputs are never scored as
completed answers. GPU/library differences can prevent bitwise reproducibility.
Back up output separately: Duke xtmp is not backed up. Data, outputs, caches,
virtual environments, and key files are gitignored.

Overrides for a separate smaller run:

```bash
HUMANS_PER_QUESTION=5 SAMPLES=1 DATA_DIR=data/sas-smoke RUN_DIR=runs/sas-smoke \
    sbatch scripts/baseline.sbatch
RUN_DIR=runs/sas-smoke sbatch scripts/score.sbatch
```

Submit the second command after the first succeeds. Other generation overrides:
`MAX_NEW_TOKENS` and `PRECISION` (`4bit` or `16bit`; 16-bit requires a 48 GB card).
For direct steps inside the appropriate compute allocation:

```bash
python -m assessengine prepare
python -m assessengine generate
python -m assessengine.snapjudge export
python -m assessengine.snapjudge run
```

## Diagnostics and tests (local CPU)

No models or detector calls:

```bash
python3 -m venv .venv-diagnostic
source .venv-diagnostic/bin/activate
python -m pip install -r requirements-diagnostic.txt
HF_HOME="$PWD/.cache/huggingface" python -m assessengine.diagnose
python -m unittest discover -s tests -v
```

The diagnostic now defaults to **all ten SAS prompts**, removing exact within-prompt
text duplicates and recording the dataset revision. It writes
`data/diagnostics/sas/report.json` and `questions.csv`. Use `--dataset mohler` for
historical Mohler analysis. Word counts use whitespace; SnapJudge is authoritative
about its own eligibility. `bash -n scripts/baseline.sbatch` and
`bash -n scripts/score.sbatch` check shell syntax.

## Interpreting results

SnapJudge requires at least 50 words and was calibrated on research papers.
Its `ai_generated_share` is the fraction of windows classified AI-generated,
**not an authorship probability**. AI-polished windows count as human-written in
the website verdict. Short answers have few windows, so scores may be coarse.
No Desklib threshold is applied. Missing/errors/exclusions are shown separately.
This is a detector baseline, not proof of genuine human–LLM divergence or cheating.
Human responses to modified prompts are still needed for the eventual research claim.

## Sources

- [ASAP-SAS original competition](https://www.kaggle.com/competitions/asap-sas)
- [S-GRADES Hugging Face conversion](https://huggingface.co/datasets/nlpatunt/D_ASAP-SAS)
- [Prompt/source documents](https://huggingface.co/datasets/nlpatunt/dataset_rubrics)
- [DeepSeek model card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)
- [SnapJudge official batch guide and client](https://nlplab.cs.duke.edu/snapjudge/batch/)

Consult original dataset terms before redistribution. Hugging Face availability
alone is not a redistribution license.
