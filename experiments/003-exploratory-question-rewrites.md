# Exploratory round: minimal-instruction Gemini rewriting

**Completed locally:** five calls to Gemini 3.1 Pro produced one full-question
rewrite per SAS prompt. All are saved in `runs/gemini-exploratory/review.md`.
The model mainly chose exaggerated technical/academic register and stylistic
mirroring. Its rationales make strong predictions, which are not experimental
results. No answer generation or detector comparison has run. Task equivalence
and reading-difficulty preservation remain unverified.

The latest protocol gives Gemini 3.1 Pro each complete SAS question and a short
instruction: preserve its meaning while freely rewriting it so an LLM may give
a more confused, mistaken, mechanical or easily detected answer. One output per
question, five calls total, no feedback loop. This is distinct from the previous
paper-guided and task-contract rounds, which remain saved.

The exact instruction is [prompts/exploratory-rewriter.md](../prompts/exploratory-rewriter.md).
No paper brief, grammar checklist, edit-distance rule, self-check contract, word
limit, or maxOutputTokens setting is sent. Provider limits still apply. The
response schema asks only for the rewritten question, rationale, and a meaning
preservation note. No historical human answers or detector results are supplied.

The full prompt, including source context, is editable in this round. There is
no automatic assertion that facts, required knowledge, or answer demands survived
the edit. A model statement of preserved meaning is not independent verification.
Changed numerical data, extra computations, new response restrictions, or greater
difficulty must be identified before calling an edit task-equivalent.

**Incorrectness and detectability are separate outcomes.** A lower correctness
rate does not demonstrate higher detector scores; a detector might flag correct
model answers or miss incorrect ones. This exploratory protocol also differs
from the original minimal task-preserving perturbation objective if its outputs
add constraints or alter answer requirements.

Artifacts are under `runs/gemini-exploratory/`: exact requests, raw responses,
original questions, proposals, review.md and run-summary.json. Keys and results
are gitignored. The configured project is EdAI Research Project, model
gemini-3.1-pro-preview. The same private .env.gemini supplies its key.

```bash
source .env.gemini
python -m assessengine.exploratory prepare
python -m assessengine.exploratory generate
cat runs/gemini-exploratory/review.md
```

These steps are being run locally for this round. Do not repeat successful calls
on the cluster. Cached responses prevent repeated calls in the same directory.
The Slurm script scripts/modify.sbatch now runs this exploratory protocol when
needed. No DeepSeek answer generation or detector submissions occur here.

Previous paper-guided protocol: [experiment 002](002-gemini-question-edits.md).

## Cluster evaluation

The exact five full Gemini outputs are frozen in
[`inputs/gemini-exploratory-v1.jsonl`](inputs/gemini-exploratory-v1.jsonl), so Git
transfers them even though `runs/` is ignored. These include Gemini's rewritten
background for sets 1 and 2, rather than the display versions assembled with the
original background. Equivalence remains unverified.

The GPU script generates **20 accepted answers per question (100 total)** using
the same pinned DeepSeek revision, quantization, sampling settings, seed scheme,
80–120-word instruction, and minimum 50-word acceptance rule as the baseline.
Up to five attempts are allowed per answer slot. Inputs go into
`data/sas-exploratory-v1/`, outputs into `runs/sas-exploratory-v1/`; the original
baseline is untouched. No historical humans are attached to rewritten prompts.

After pushing local changes and pulling on the cluster:

```bash
cd /usr/xtmp/ms785/AssessEngine
git pull --rebase origin main
module load python/3.12.8
mkdir -p logs
sbatch scripts/rewrite-baseline.sbatch
```

Allow up to four hours of GPU runtime, plus queue time. If it times out, submit
the same script again; saved accepted answers are reused. Once generation has
completed successfully, run:

```bash
cat runs/sas-exploratory-v1/generation_coverage.json
RUN_DIR=runs/sas-exploratory-v1 sbatch scripts/score.sbatch
```

Require `all_quotas_met: true` before scoring. No `ALLOW_PARTIAL` flag is used.
The scoring job uses the existing `.secrets/snapjudge.key` and submits 100 new
AI answers, with no repeated baseline submissions. Results include binary
verdicts, raw continuous score margins, and per-answer records:

```bash
cat runs/sas-exploratory-v1/snapjudge/report.json
```

For the original baseline, recompute the new report fields from saved results
locally on the cluster, with no API calls:

```bash
.venv/bin/python -m assessengine.snapjudge report --run runs/sas-balanced
```

Compare AI detection rates and continuous margins **within each question**.
The new run has 20 answers per question; the historical run has 50 for four
questions and 21 for set 10, so pooled rates have different question weights.
This is a historical-baseline comparison, not a simultaneous controlled rerun.
Check detector versions/thresholds in the saved raw responses before comparing
raw margins. A rise in scores alone establishes neither preserved meaning nor
answer correctness. Continuous margins are uncalibrated confidence proxies.
