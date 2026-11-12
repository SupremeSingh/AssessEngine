# Experiment 002: Can Gemini modify a question to increase detector confidence?

## Active protocol: one shot, no feedback

**Run completed locally:** five successful Gemini 3.1 Pro calls produced five
rewrites, all passing the local checks. The review is saved in
`runs/gemini-one-shot/review.md`, and ten original/modified conditions have been
materialized in `runs/gemini-one-shot/evaluation-data/questions.jsonl`. No
DeepSeek answers or SnapJudge scores have been generated for these conditions.
`run-summary.json` records the actual returned model version and call counts.
Local checks plus model self-assessment are not proof of meaning preservation.

The active run is **runs/gemini-one-shot/** using EdAI Research Project and
**Gemini 3.1 Pro Preview**. It uses exactly one generation call per question,
five calls total, and returns exactly one final rewrite for each question.
There are no detector scores or target-model outputs in the request, no
candidate search, no second judge call, and no feedback/revision loop.

Each response includes a short task contract, a counterexample-based self-check,
an explicit self-assessed equivalence claim, and an answer-level linguistic
hypothesis. These are concise audit outputs, not independent semantic validation.
Gemini checks and finalizes its single output in one call. All required knowledge,
counts, scope, evidence, uncertainty and response format must remain unchanged;
context, tables and options are frozen. Small edits are allowed if their hypothesis
is defensible; visual edit size is not the objective.

```bash
# These commands run locally; already saved successful calls are reused.
source .env.gemini
python -m assessengine.gemini prepare
python -m assessengine.gemini generate
cat runs/gemini-one-shot/review.md
```

The Slurm alternative is `sbatch scripts/modify.sbatch`; do not run it again
to repeat a completed local run. The key and output files are gitignored, so
completed conditions need a separate transfer before cluster answer generation.
If all candidates pass local checks, `python -m assessengine.gemini materialize`
creates ten original/modified conditions under the active run directory. It
does not generate DeepSeek answers or call SnapJudge. Flags are retained rather
than silently repaired through additional model calls.

The next experiment compares fresh DeepSeek responses for originals and rewrites
under the same settings. It measures continuous SnapJudge decision margins and
binary detection. Historical student answers are not responses to the new prompts.
Earlier sections below describe prototype rounds; feedback optimization described
in the separate research note is **not** part of this one-shot protocol.

## Revised instruction round: answer-elicitation hypotheses

The first five rewrites were largely cosmetic. A second round now asks Gemini
to use the paper as suggestions for naturally eliciting more information-dense,
noun-heavy model answers, rather than simply replacing synonyms in the question.
It explicitly requires the chain: **question framing → predicted answer
construction → paper feature → uncertain detector effect**.

Suggested mechanisms include foregrounding already-required entities/properties,
framing existing information requests around the items and their descriptions,
and reorganizing already-required subparts around a shared process. Nominalizations,
subject-position that-relative clauses and participial postnominal clauses motivate
these suggestions. None is a demonstrated prompt intervention from the paper.
No new knowledge, example, justification, response format or persona can be added.
The respondent-facing prompt cannot tell the model to sound AI-like or use features.

Latest output directory: **runs/gemini-edai-style-v2/**. Earlier proposals remain
in **runs/gemini-edai-edits/**. The default CLI and Slurm output directory now
point to the revised round. The run hashes its exact instruction files so the
two rounds cannot be silently combined.

Character similarity is no longer treated as a semantic guarantee or rejection
rule. Reordering wording and removing numbered task labels is allowed; substantive
counts still must match. The local screen preserves “at least” and flags replacing
tentative “might” with definite wording. It rejects excessive expansion, obvious
authorship/format cues and unchanged tasks. Passing remains a preliminary screen,
not a proof of task equivalence. Inspect the actual requirements in each rewrite.

The revised run completed with five proposals. Sets 1, 2, 5 and 6 passed local
checks. Set 10 was flagged because it changed tentative “might affect” into
effects the color “will produce”; it has not been silently accepted. No new
DeepSeek responses or detector comparisons have been run.

Read **runs/gemini-edai-style-v2/review.md**. Materializing
with the default directory creates ten original/modified conditions and makes no
DeepSeek or detector calls. The instructions below describing the earlier round
are retained as history; use the latest directory for new work.

**Status: five Gemini 3.1 Pro rewrites generated successfully using EdAI Research
Project. Two pass local checks; sets 1, 2 and 10 are flagged for inspection. No
DeepSeek generation or detector evaluation of these modified questions has run.** Here “confidence” means a continuous detector **decision score**,
not a calibrated probability that an answer was written by AI.

## Simple experiment

1. Give Gemini each of our five SAS questions and suggestions grounded in the paper.
2. Ask for exactly **one meaning-preserving rewrite per question**.
3. Generate fresh DeepSeek answers to the original and modified questions using
   the same checkpoint, length instruction, sampling and retry policy.
4. Compare the mean continuous detector score and binary detection rate per question.

There is **no paraphrase control, best-of-N search, or separate Gemini judge**.
Five questions produce five rewrites and ten original/modified evaluation conditions.
Gemini supplies a rationale for preserved meaning; local checks flag obvious drift.
Those checks and Gemini's own rationale cannot prove semantic or educational validity.
Inspect the proposed questions before interpreting the eventual results.

## Paper as suggestions

The model receives [a paper context brief](../prompts/style-paper-context.md)
and [simple modifier instructions](../prompts/question-modifier.md). The brief
explains noun-heavy/information-dense patterns, relevant Biber constructions,
the importance of genre/model differences, and the limitations of transfer.
It does not prescribe grammatical constructions or require Gemini to force them
into answers. The paper does not itself show that these edits improve detection.
Gemini is not given historical student answers, detector scores, or missed examples.

The original context, tables and set-10 color choices remain unchanged. Only
existing task wording can change. Required counts, scientific scope, knowledge,
starting points, evidence and response format must remain the same. The shared
80–120-word final-answer instruction stays outside the editable question.

## Powerful Gemini model and project

Default: **gemini-3.1-pro-preview**, the Pro-series model documented by Google.
The client checks that this model supports generateContent for your key; it does
not silently fall back to Flash. Actual returned modelVersion and token usage
are saved. This is a preview model and availability/billing depend on the project.

- Project: EdAI Research Project
- Project name: projects/861354941617
- Project number: 861354941617
- Project ID: edai-research-project

Create an API key under this project in Google AI Studio. These public IDs are
not authentication. Requests use the Gemini Developer API with x-goog-api-key,
not Vertex AI. The client records the intended project but cannot verify a
key's project association; check that association in AI Studio.

## Run the modification stage

After pushing changes, on the cluster:

```bash
cd /usr/xtmp/ms785/AssessEngine
git pull
module load python/3.12.8
source .venv/bin/activate
cp -n config/gemini.env.example .env.gemini
chmod 600 .env.gemini
nano .env.gemini
```

Fill the private key at **# Add Key Here** and make sure the model line is:

```bash
export GEMINI_MODEL="gemini-3.1-pro-preview"
```

If an earlier .env.gemini exists, update its model line: cp -n does not replace it.
Do not put real keys in the committed template or chat. Then:

```bash
mkdir -p logs
sbatch scripts/modify.sbatch
```

The job prepares five requests and makes **five generation calls total**, one per
question. No GPU, detector calls or new SDK is needed; requests is already installed.
Completed calls are cached. An interrupted call with no saved response leaves a
pending marker so it cannot be silently repeated. Pricing depends on your AI Studio
project; this code does not promise free Gemini access.

Output directory: **runs/gemini-edai-edits/**, separate from the superseded
multi-variant prototype. For offline preparation only:

```bash
source .env.gemini
python -m assessengine.gemini prepare
```

This makes no API calls. The ai-studio/set-*.md files are also copy/paste prompts
for manual UI exploration; the API path records the reproducible run.

After the job:

```bash
cat runs/gemini-edai-edits/review.md
python -m assessengine.gemini materialize
```

Materialization requires all five rewrites to pass local checks. It produces
**runs/gemini-edai-edits/evaluation-data/questions.jsonl** with ten conditions:
original and paper_inspired for each question. No historical human answers are
attached to the modified conditions. It does not run DeepSeek or a detector.
All raw Gemini request/response JSON, returned model versions, hypotheses,
rationales and local-check results are retained. The context and submitted
candidate cannot be edited after validation without being detected.

## What confidence data do we have?

Our original baseline summary reported **binary website verdicts** and generated
window share. All 442 answers had one window, so the shares were also binary
(0 or 1). They are not graded confidence measurements.

The saved raw SnapJudge JSON also contains continuous three-class logit-derived
scores: **nonhuman** and **gen_minus_pol**, plus the corresponding thresholds.
We now expose them in scores.jsonl and continuous_scores.csv. No new API calls
are needed to recover these scores from existing results:

```bash
python -m assessengine.snapjudge report
```

The derived **generated decision margin** for a window is:

```text
min(nonhuman - thr_nonhuman, gen_minus_pol - thr_generated)
```

For a single-window answer, a positive value puts both scores beyond the
generated-class thresholds. Higher values mean further inside that decision
region. Negative values fail at least one threshold. This is an **uncalibrated
signed decision score**, not a percentage confidence or probability. Do not
apply a softmax and present it as validated authorship confidence.

For the 442 archived baseline answers, this sign agrees with all website binary
verdicts. Mean scores are **-21.1188 for humans** and **+3.0215 for AI answers**.
All documents had one window. For multi-window texts, the code reports a mean
window margin, which is not equivalent to the document's fraction-based verdict.
Missing or unsupported score types are reported as unavailable rather than invented.

## Comparison to run next

Primary question: is the mean generated decision margin higher under the rewritten
question than under the original? Report the difference per question, the answer
counts and distribution, and binary detection rates as a secondary outcome.
Use the same detector score type, model and thresholds. If they change between
runs, the margins cannot be treated as a clean before/after comparison.

Use **fresh original answers alongside fresh modified answers** to avoid comparing
only to a historical run. Twenty qualifying answers per condition would mean
200 answers across five questions and two conditions. Reuse DeepSeek revision
1df8507178afcc1bef68cd8c393f61a886323761 and the same generation settings.
Do not call higher detector confidence proof of better assessment or genuine
human–LLM divergence. Modified questions will eventually need new human answers.
No added paraphrase condition is included in this simple pilot, so a gain cannot
be attributed specifically to the paper's advice rather than rewording generally.

## Sources

- [Paper](https://aclanthology.org/2026.findings-emnlp.1375/)
- [Gemini 3.1 Pro Preview](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-pro-preview)
- [Structured outputs](https://ai.google.dev/gemini-api/docs/generate-content/structured-output)
- [AI Studio API key setup](https://ai.google.dev/gemini-api/docs/api-key)
