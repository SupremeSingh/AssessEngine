# AssessEngine

## Experments

**Can SnapJudge detect DeepSeek-generated answers to educational questions?**

In our five-question pilot, SnapJudge detected **206 of 221 AI answers (93.2%)**
and flagged **none of the 221 human answers**, giving **96.6% accuracy** on this sample.

This is a promising baseline for selected answers of at least 50 words, with
most misses concentrated in the protein-synthesis question. Results cover five
selected questions and do not establish generalization to other questions or
models. [Detailed experiment, tables, and limitations](experiments/001-snapjudge-sas-baseline.md).

**Can Gemini rewrite questions so SnapJudge detects more DeepSeek answers?**

In our exploratory run, SnapJudge detected **214 of 221 rewritten-question
answers (96.8%)**, compared with **206 of 221 original-question answers (93.2%)**.
That is **8 more detections, a 3.6 percentage-point increase**.

Gemini 3.1 Pro produced one rewrite per question with minimal instructions and
no detector feedback. The counts match within question, but meaning preservation
and answer correctness remain unverified; this aggregate comparison does not
establish statistical significance or improved human–AI separation on rewritten
prompts. [Results, score interpretation, and limitations](experiments/004-snapjudge-rewrite-results.md).

The tested rewrites include changes to source context for sets 1 and 2.
[Exact protocol and cluster commands](experiments/003-exploratory-question-rewrites.md).

The frozen rewrites are committed under `experiments/inputs/`; the completed
evaluation used a separate run directory, preserving the original baseline.
Earlier paper-guided proposals were not the rewrites evaluated in this run.

## Active workflow: SnapJudge (supersedes Pangram instructions below)

Pangram API scoring is stopped. The default `scripts/score.sbatch` now uses
**SnapJudge's official online batch client** with `--no-save`, concurrency 3,
and automatic client retries/resumption. Pangram files and its ledger are
preserved, but no Pangram endpoint or key is used by the job.

Use the saved **221 human and 221 completed AI answers**, matched by count within
question: 50 each for sets 1, 2, 5, 6, and 21 each for set 10. `--allow-partial`
uses the same detector-independent selection for both adapters. Human selection
uses the original deterministic sample order, never detector scores. All unused
humans and rejected generations remain untouched. Matching counts is not pairing
individual students with AI answers. No further generation is needed.

The supplied SnapJudge key is saved locally in **`.secrets/snapjudge.key`**, which
is gitignored and private. Do not put it in source, chat, shell arguments, or logs.
From your **Mac terminal**, copy the private file separately from Git:

```bash
scp /Users/manmit/Desktop/AssessEngine/.secrets/snapjudge.key \
  ms785@login.cs.duke.edu:/usr/xtmp/ms785/AssessEngine/snapjudge.key
```

After pushing code changes, on the **cluster**:

```bash
cd /usr/xtmp/ms785/AssessEngine
git pull
mkdir -p .secrets logs
chmod 700 .secrets
mv snapjudge.key .secrets/snapjudge.key
chmod 600 .secrets/snapjudge.key
module load python/3.12.8
source .venv/bin/activate
python -m assessengine.snapjudge export --allow-partial
ALLOW_PARTIAL=1 sbatch scripts/score.sbatch
```

Export should show `human:eligible: 221` and `ai:eligible: 221`. It spends no
detector requests. The scoring job sends one text document per answer, with no
questions, grading rubrics, labels, or reasoning in submitted text. Only the
key file's path is passed to the official client. The script is cached locally,
with URL and SHA-256 recorded. This local key has not been used to submit the
cluster answers from development; real remote operation remains to be verified.

```bash
tail -f logs/assess-score-JOB_ID.out
cat runs/sas-balanced/snapjudge/report.json
```

If SnapJudge finished all documents but reporting failed with `KeyError: 'file'`,
pull the parser fix and rebuild **locally from saved results**, without submitting
anything again:

```bash
module load python/3.12.8
source .venv/bin/activate
python -m assessengine.snapjudge report
```

The official CSV has a UTF-8 byte-order mark; the parser now accepts it as well
as ordinary UTF-8. No new job, API key, or detector requests are needed for reporting.

Results are kept separately under `runs/sas-balanced/snapjudge/`: `results.csv`,
`results_json/` (full raw responses), `scores.jsonl` joined with authorship,
`index.jsonl`, `export.json` (input hashes/selection policy), `coverage.json`,
and `report.json`. Re-submit the same command to resume finished documents.
Do not run simultaneous writers or change inputs after exporting.

The service requires **50 words per document**, and was calibrated on research
papers. Its generated-window share is not an authorship probability; polished
windows count as human-written in the website verdict. Short answers have few
windows. `--no-save` requests no retention of submissions under the service's
documented policy. API keys/quotas for SnapJudge are distinct from Pangram credits;
Pangram's 1,500-credit policy does not describe SnapJudge billing.

[Official SnapJudge guide/client](https://nlplab.cs.duke.edu/snapjudge/batch/).
Tests use fake responses only. Run `python -m unittest discover -s tests -v`
and `bash -n scripts/score.sbatch` for local checks. The sections below document
the previous Pangram experiment and dataset preparation for reproducibility;
their Pangram scoring commands are **not the current run instructions**.

ASAP-SAS science/biology questions, authentic student answers, fresh DeepSeek
responses, and **Pangram realtime detection**. Pangram replaces SnapJudge in the
default scoring workflow. The ongoing generation does not need changing.

## Current experiment

Five prompts (SAS sets **1, 2, 5, 6, 10**), **250 human answers** (50 per prompt),
and **250 qualifying DeepSeek-R1-Distill-Qwen-14B answers**. Raw human answers are
sampled deterministically, exact-text-deduplicated, and at least 50 words. AI
prompts request 80–120 words; each answer slot has at most five independent
attempts. Only completed final answers of at least 50 whitespace-counted words
qualify. All attempts, reasoning, seeds, timing and rejection counts are saved.
By default scoring requires every quota to be met. The explicit partial-run
workflow below can instead use balanced saved answers without further generation.

## Score the saved answers after the timeout (current workflow)

Generation job 13550364 stopped at the time limit with **221 qualifying AI answers**:
50 on each of sets 1, 2, 5, 6 and 21 on set 10. No more generation is needed.
Use **221 human answers**, matching counts within each prompt, for **442 documents**.
This is count matching within a question, not individual student/AI answer pairing.
The first N humans in the existing deterministic sampled order are used; selection
does not depend on detector results. All source data, rejected attempts, and the
29 unused human answers remain intact.

After pushing the local changes, run on the cluster:

```bash
cd /usr/xtmp/ms785/AssessEngine
git pull
module load python/3.12.8
source .venv/bin/activate
python -m assessengine.pangram plan --allow-partial
```

Check `documents: 442`, `by_authorship: {human: 221, ai: 221}`,
the per-question counts, and `within_budget: true`. The preview spends no credits.
Set the Pangram key using the section below, then:

```bash
ALLOW_PARTIAL=1 sbatch scripts/score.sbatch
```

This does not run DeepSeek. Selection IDs/hashes are recorded in
`runs/sas-balanced/pangram/selection.jsonl`. The selection policy is included in
the run fingerprint; it cannot be silently changed after scoring begins.
The 1,500-credit estimated spending ceiling and 1,500-credit reserve still apply.
Do not resume generation or edit the source files during this scoring experiment.
Rebuilding this partial report requires
`python -m assessengine.pangram report --allow-partial`.

## After the running generation finishes

Wait for the current job to finish **before pulling workflow changes**. Push the
local changes to Git, then on Duke:

```bash
cd /usr/xtmp/ms785/AssessEngine
git pull
module load python/3.12.8
source .venv/bin/activate
cat runs/sas-balanced/generation_coverage.json
python -m assessengine.pangram plan
```

`all_quotas_met` must be true. **`plan` makes no API calls and spends no credits.**
It writes `runs/sas-balanced/pangram/plan.json` with counts, estimated credits,
cache reuse, and budget checks. Confirm `within_budget: true` before scoring.

## Add the API key

```bash
cp -n config/pangram.env.example .env.pangram
chmod 600 .env.pangram
nano .env.pangram
```

Fill the private copy at the **Add Key Here** comment:

```bash
# Add Key Here
export PANGRAM_API_KEY="your-key"
```

Never put a real key in the committed template, chat, command arguments, or job
logs. `.env.pangram` is gitignored and automatically sourced by the scoring job.
An exported `PANGRAM_API_KEY` also works. A website subscription does not necessarily
supply API credits; live API balance/model checks must succeed.

Then submit one CPU job:

```bash
mkdir -p logs
sbatch scripts/score.sbatch
```

It uses `requests` in the existing `.venv`, sends texts **one at a time** with
`model="pangram-4"`, `public_dashboard_link=false`, and polls saved task IDs.
It sends answer text only: no labels, questions, rubrics, or rejected attempts.
No GPU, new SDK, or further generation is needed. No paid calls have been made
from development; tests use mocked API responses.

```bash
tail -f logs/assess-score-JOB_ID.out
cat runs/sas-balanced/pangram/report.json
```

Direct scoring inside a compute allocation:

```bash
source .env.pangram
python -m assessengine.pangram run
```

## Budget: 3,000 credits

### Recovery when the balance API returns 404

The cluster returned HTTP 404 for `GET /balance` before any text was submitted.
This deployment may not expose the documented account endpoint. After pulling
the fix, explicitly opt into local estimated-credit budgeting:

```bash
ALLOW_PARTIAL=1 ALLOW_MISSING_BALANCE=1 sbatch scripts/score.sbatch
```

This bypasses only a 404 from the balance read, not authentication errors,
rate limits, other server failures, or an insufficient live balance when available.
The 1,500 estimated-credit ceiling, shared ledger and duplicate-charge protections
remain active. It cannot verify the actual account balance/reserve when that API
is absent; check the Pangram dashboard and avoid other account usage. The current
442-text preview estimates 503 credits. Reservations record which balance guard
was used. Direct CLI equivalent: add `--allow-missing-balance --allow-partial`.

Defaults cap the baseline at **1,500 estimated credits**, reserving **1,500**.
Each unique text estimates `ceil(whitespace_words / 100)` credits, minimum one,
at $0.05 per credit. Thus 80 words estimate one credit and 120 estimate two.
Provider word counting and billing are authoritative; these are conservative
rounding estimates, not a guaranteed account-wide spending cap.

Before every new paid POST, the free `/balance` endpoint must show sufficient
USD to preserve the reserve ($75 at the documented rate), plus a 10% cushion
on the next estimated charge. An unexpected balance representation stops the
job; do not disable the guard before checking your account's API credits.

The shared `.cache/pangram/ledger.json` records estimated reservations, task IDs,
and complete results. Identical text/model hashes reuse results across runs,
including repeated human texts. Preserve and back up this ledger: deleting or
changing its path resets local spending history. A lock prevents simultaneous
jobs sharing the ledger. Other machines, manual activity, provider billing
changes, and delayed balances remain outside this local policy; use Pangram
account controls for authoritative limits and avoid concurrent account use.

No automatic paid retries. A failed task stops scoring. A timeout after POST
can still mean a billed task: a durable `submitting` record without a task ID
blocks automatic resubmission. Reconcile it with Pangram before changing it.
Polling failures/timeouts resume the saved task ID; completed tasks are reused.
Re-submit the same scoring command to resume after a polling interruption.

Budget overrides are explicit: `--max-credits`, `--total-credits`, and
`--reserve-credits`. Inspect the preview before changing any. Bulk would be
cheaper but is intentionally not used in this sequential pilot.

## Outputs

Under `runs/sas-balanced/pangram/`:

- `plan.json`: counts, credit estimates, reservations, budget checks.
- `selection.jsonl`: selected answer IDs, authorship, word counts and content hashes.
- `inputs.json`: model and exact input file hashes.
- `scores.jsonl`: answer/question IDs, authorship, text, task IDs, state, full raw results.
- `scores.csv`: classifications, fractions, actual model versions, and metadata.
- `report.json`: completed/failed counts, verdicts, mean fractions by authorship,
  and returned version counts. Rebuild with `python -m assessengine.pangram report`.

Raw responses retain segments/windows when returned. `fraction_ai`,
`fraction_ai_assisted`, and `fraction_human` describe portions of text, not
probabilities of authorship. AI assistance stays separate. `pangram-4` can serve
different minor versions; actual returned versions are recorded. Historical
SnapJudge/Desklib results are preserved and never mixed into Pangram outputs.

## New generation jobs and timeout recovery

```bash
module load python/3.12.8
export PYTHON_BIN="$(command -v python3)"
export HF_HOME="$PWD/.cache/huggingface"
mkdir -p logs
sbatch scripts/baseline.sbatch
```

One A5000, four CPUs, 48 GB host memory, four-hour limit. After generation, only
an **offline Pangram plan** runs; the job never spends detector credits. Resume
after a timeout with the same command, never while another generation is active.
Use `--gres=gpu:a6000:1` for an A6000. Python 3.10–3.12, a compatible CUDA driver,
internet on first install, and about 60 GB disk are required. Full model weights
are downloaded even for 4-bit loading. Hugging Face login may be needed for rate limits.

Defaults: `data/sas`, `runs/sas-balanced`. Old `runs/sas-baseline`/Mohler results
are untouched. Changed generation configurations are rejected on resume.
`DATA_DIR`, `RUN_DIR`, `HUMANS_PER_QUESTION`, `SAMPLES`, `MAX_ATTEMPTS`,
`MAX_NEW_TOKENS`, and `PRECISION` configure separate runs. `RUN_DIR` also selects
another scoring run. Accepted slots are not regenerated; rejected attempts
count against the retry budget. GPU/library differences can affect reproducibility.

## Dataset integrity and interpretation

The authentic pre-ChatGPT ASAP-SAS mirror is pinned at
`nlpatunt/D_ASAP-SAS`, revision `f97ad6415c94c763b901423b72f78f1fbf46d630`.
It removes grades; we do not use grades or supply human answers to DeepSeek.
Sets 5 and 6 contain rubrics in the prompt field: these are removed at explicit
boundaries, failing on unexpected structure. Sets 1, 2, and 10 retain the mirror's
procedures/tables. Source URLs, hashes and optional image URLs are saved. Set 10
uses the mirror's text transcription, not vision; not every original image has
been independently checked. Eligible pools are 696, 849, 194, 206, 506 respectively.

We pool answer-level source splits for this descriptive pilot, retaining IDs/splits.
This does not create held-out question families. Reserve prompt families before
Gemini optimization. Five science prompts cannot establish broad CS transfer.
Selecting long humans, adding AI-only length instructions, and retrying short AI
answers condition the sampled distributions. Report rejection rates, actual
lengths and correctness alongside detector performance. Modified prompts still
need newly collected human answers to establish true human–LLM divergence.

## Local diagnostics and tests

```bash
python3 -m venv .venv-diagnostic
source .venv-diagnostic/bin/activate
python -m pip install -r requirements-diagnostic.txt
HF_HOME="$PWD/.cache/huggingface" python -m assessengine.diagnose
python -m unittest discover -s tests -v
bash -n scripts/baseline.sbatch
bash -n scripts/score.sbatch
```

The diagnostic covers all ten SAS prompts, writing `data/diagnostics/sas/`.
Use `--dataset mohler` for historical analysis. Tests spend no detector credits.
Data, runs, keys, caches, and environments are gitignored. Duke xtmp is not
backed up; preserve results and the Pangram ledger elsewhere.

## Sources

- [ASAP-SAS original](https://www.kaggle.com/competitions/asap-sas) and
  [mirror](https://huggingface.co/datasets/nlpatunt/D_ASAP-SAS).
- [Prompt sources](https://huggingface.co/datasets/nlpatunt/dataset_rubrics).
- [DeepSeek model](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B).
- [Pangram API](https://docs.pangram.com/api-reference/introduction),
  [balance](https://docs.pangram.com/api-reference/balance),
  [models](https://docs.pangram.com/api-reference/models),
  [billing units](https://www.pangram.com/contact-us/research).

Consult original dataset terms before redistribution. SnapJudge's legacy adapter
remains for reproducibility but is not called by either job script.
