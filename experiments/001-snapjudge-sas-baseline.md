# Experiment 001: Can SnapJudge detect DeepSeek answers to educational questions?

**Date:** November 11, 2026. **Status:** completed descriptive baseline; no prompt optimization.

SnapJudge classified 206 of 221 freshly generated DeepSeek answers as AI-generated
and all 221 authentic student answers as human-written. Under its website verdict
policy, this is **93.21% AI recall, 0% observed human false positives, and 96.61%
accuracy** on the selected sample. Five additional AI answers received the raw
class *polished*, which the website counts as human-written; ten received raw
class *human*.

These results describe five science/biology prompts and answers of at least
50 words. They do not establish general detector accuracy, educational validity,
cheating detection, or increased human–LLM divergence under modified prompts.

## 1. Research question and design

**Question:** Can the online SnapJudge text detector distinguish authentic student
answers from freshly generated DeepSeek answers to the same educational prompts?

This experiment establishes the original-prompt detector baseline before asking
Gemini to modify questions. The answering model is frozen; no detector, modifier,
LoRA adapter, or reinforcement-learning policy was trained. The only added model
instruction specifies final-answer length and explanatory style. No question
modifications were tested in this experiment.

The unit scored is one answer, not a question or a concatenation of answers.
The groups are matched by **answer count within question**, not by individual
student, ability, answer correctness, or exact length.

## 2. Dataset and human-answer selection

We used [ASAP-SAS](https://www.kaggle.com/competitions/asap-sas), a pre-ChatGPT
student-response collection, via the [S-GRADES Hugging Face mirror](https://huggingface.co/datasets/nlpatunt/D_ASAP-SAS).
The mirror's grade labels were not used. We selected science/biology sets
**1, 2, 5, 6, and 10**. This is not a computer-science assessment dataset.

| Set | Assessment task | Loaded answers | Unique exact texts | Eligible: ≥50 words | Initially sampled | Finally scored |
|---|---|---:|---:|---:|---:|---:|
| 1 | Identify information needed to replicate a vinegar/materials experiment | 1,672 | 1,672 | 696 | 50 | 50 |
| 2 | Interpret plastic stretchability and improve experimental design | 1,278 | 1,277 | 849 | 50 | 50 |
| 5 | Describe four major steps of protein synthesis | 1,795 | 1,785 | 194 | 50 | 50 |
| 6 | Describe three cell-membrane transport processes | 1,797 | 1,791 | 206 | 50 | 50 |
| 10 | Use lid-color/temperature data to recommend doghouse paint | 1,640 | 1,639 | 506 | 50 | 21 |
| **Total** | **Five prompts** | **8,182** | **8,164** | **2,451** | **250** | **221** |

Duplicate removal was by exact answer text within a prompt. Eligibility used
whitespace-separated word counts. Raw human answers were not corrected,
expanded, paraphrased, padded, or combined. Sampling used seed 42 with a
per-set offset. Source response IDs and train/validation/test memberships were
retained. The source answer-level partitions were pooled for this descriptive
pilot; there is no held-out question-family evaluation here.

The initial human sample had 50 answers per prompt. After generation timed out,
the final sample used the first N answers in that existing deterministic sample
order, where N was the number of qualifying AI answers for that prompt. Thus
29 selected human answers from set 10 remain saved but were not scored. Selection
did not use detector outcomes.

### Prompt preparation

The mirror's set-5 and set-6 prompt fields contain grading rubrics. These were
removed at explicit rubric boundaries; the set-5 item title was also removed.
Experiment procedures and tables were retained for sets 1, 2, and 10. The model
received no student answers, grades, reference answers, or rubrics.

Set 10 includes a source image URL. The text-only model used the mirror's
transcribed context and table; the transcription has not been independently
validated against every original assessment image. Source URLs, raw prompt
hashes, and processing descriptions are stored with the prepared questions.

## 3. DeepSeek generation

| Setting | Recorded value |
|---|---|
| Model | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` |
| Checkpoint revision | `1df8507178afcc1bef68cd8c393f61a886323761` |
| Hardware | NVIDIA RTX A5000 for every saved attempt |
| Precision | NF4 4-bit loading; double quantization |
| Seed | 42, expanded deterministically by question, answer slot, and attempt |
| Sampling | Temperature 0.6; top-p 0.95; top-k disabled |
| Generation cap | 4,096 new tokens, including reasoning |
| Requested final length | 80–120 words; explicitly at least 50 words excluding reasoning |
| Acceptance | EOS-completed, nonempty final answer after `</think>`, with ≥50 words |
| Retry limit | Five attempts per answer slot |
| Intended quota | 50 qualifying answers per prompt; 250 total |
| Realized quota | 221 qualifying answers |

The length instruction also requests complete sentences without repetition or
filler. This is an AI-only generation condition and was not an instruction given
to the historical students. The requested 80–120 range is advisory: code accepts
any completed final answer of at least 50 words, including answers above 120.

Generation processed prompts sequentially and reached the cluster job's time
limit while working on set 10. The last in-flight attempt was not a completed
saved response. Generation was then deliberately stopped in favor of evaluating
the saved answers. No retry slot exhausted its five-attempt allowance in the
saved sample.

| Set | Saved attempts | Qualifying final answers | Rejected: <50 words | Rejection rate | Missing from original quota |
|---|---:|---:|---:|---:|---:|
| 1 | 54 | 50 | 4 | 7.41% | 0 |
| 2 | 53 | 50 | 3 | 5.66% | 0 |
| 5 | 50 | 50 | 0 | 0% | 0 |
| 6 | 50 | 50 | 0 | 0% | 0 |
| 10 | 26 | 21 | 5 | 19.23% | 29 |
| **Total** | **233** | **221** | **12** | **5.15%** | **29** |

All 233 saved attempts had completion status `complete`; none of these saved
attempts was truncated or missing its reasoning boundary. The 12 rejections
were solely short final answers. Rejected attempts remain in the archive but
were not sent to the detector. Acceptance among saved attempts was **94.85%**.

The saved generations contain **130,576 generated tokens**, including reasoning
and rejected attempts. Their recorded generation durations sum to
**13,263.724 seconds (3.684 hours)**. This excludes the unfinished attempt,
model loading, setup, and scheduling; it is not the Slurm job's full runtime.

## 4. SnapJudge execution and score semantics

We used the [official SnapJudge online batch client](https://nlplab.cs.duke.edu/snapjudge/batch/),
with concurrency **3** and **`--no-save`**. Each eligible answer was submitted as
its own UTF-8 text document. Question text, local authorship labels, and DeepSeek
reasoning were not part of the submitted document. The key is not included in
this report or the data archive.

| Execution measure | Result |
|---|---:|
| Submitted documents | 442 |
| Human / AI documents | 221 / 221 |
| Successful responses | 442 |
| API errors / missing results | 0 / 0 |
| Analysis windows per document | Exactly 1 for all 442 documents |
| Raw result `saved_for_research` | `false` for every document |
| First / last client timestamps | `2026-11-11T19:37:05` / `2026-11-11T19:37:38` |
| Span between recorded timestamps | 33 seconds |
| Sum of per-document client durations | 93.6 seconds; concurrent, not wall-clock time |

The client timestamps have no timezone suffix; this report does not infer their
timezone. A local CSV parser initially failed on the official client's UTF-8
byte-order mark after detection finished. The parser was corrected and the
report rebuilt from saved results, without repeating detector submissions.

### Binary website verdict versus raw three-class result

SnapJudge's website counts *polished* text as human-written. We used that policy
for the primary binary metrics and retained the raw class separately.

| Known source | Raw human | Raw generated | Raw polished | Total |
|---|---:|---:|---:|---:|
| Authentic student | 221 | 0 | 0 | 221 |
| DeepSeek | 10 | 206 | 5 | 221 |

The five polished classifications are **not evidence that these answers were
actually human-edited**: their known source is fresh model generation. Combining
raw generated and polished as an alternative “nonhuman signal” would flag
**211/221 AI answers (95.48%)** with **0/221 human flags**, but that is a separate,
post hoc interpretation, not the primary website-verdict result.

With one window per answer, `ai_generated_share` is binary here: **0 or 1**.
Its group average is therefore the fraction of answers flagged, not an average
confidence probability. It should not be interpreted as a calibrated probability
of authorship.

Recorded server metadata: model `Text detector`, layer `text`, three classes,
score type `logit3`, `thr_nonhuman=11.95624`, and `thr_generated=10.59513`.
The adapter did not tune these thresholds. The response does not provide an
immutable detector checkpoint identifier; the saved client hash and raw outputs
support auditing but cannot guarantee future service-version reproducibility.

## 5. Primary results

Positive class is **AI-generated** under the website's verdict policy.

| Actual source | Predicted human-written | Predicted AI-generated | Total |
|---|---:|---:|---:|
| Authentic student | **221 (TN)** | **0 (FP)** | 221 |
| DeepSeek | **15 (FN)** | **206 (TP)** | 221 |
| **Total** | **236** | **206** | **442** |

| Metric | Calculation | Result |
|---|---|---:|
| AI recall / sensitivity | 206 / 221 | **93.21%** |
| AI false-negative rate | 15 / 221 | **6.79%** |
| Human specificity | 221 / 221 | **100% observed** |
| Human false-positive rate | 0 / 221 | **0% observed** |
| AI precision | 206 / 206 | **100%** |
| AI-class F1 | 412 / 427 | **96.49%** |
| Accuracy | 427 / 442 | **96.61%** |
| Balanced accuracy | (93.21% + 100%) / 2 | **96.61%** |
| Mean generated-window share, humans | 0 / 221 | **0.0000** |
| Mean generated-window share, AI | 206 / 221 | **0.9321** |

These are descriptive, pooled point estimates. There are only five distinct
questions and responses within a question share topic and instructions. No
independence-based significance test or confidence interval is asserted. In
particular, zero observed human false positives does not establish a zero
population false-positive rate.

### Results by question

| Set | Human n | Human false positives | AI n | AI detected | AI missed | AI recall | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 50 | 0 | 50 | 48 | 2 | 96% | 98% |
| 2 | 50 | 0 | 50 | 48 | 2 | 96% | 98% |
| 5 | 50 | 0 | 50 | 39 | 11 | **78%** | **89%** |
| 6 | 50 | 0 | 50 | 50 | 0 | 100% | 100% |
| 10 | 21 | 0 | 21 | 21 | 0 | 100% | 100% |
| **Pooled** | **221** | **0** | **221** | **206** | **15** | **93.21%** | **96.61%** |

Equal-question macro AI recall is **94.00%**; macro accuracy is **97.00%**.
These differ from pooled metrics because set 10 has fewer answers. Protein
synthesis (set 5) contributes **11/15 misses (73.33%)**, and its AI miss rate
is **22%**. This localizes the weakest baseline setting without establishing
why the detector misses it.

## 6. Answer lengths and remaining confounds

| Scored group | n | Total words | Mean | Median | Minimum | Maximum |
|---|---:|---:|---:|---:|---:|---:|
| Human | 221 | 15,523 | 70.24 | 64 | 50 | 242 |
| DeepSeek | 221 | 18,567 | 84.01 | 80 | 50 | 215 |
| **Combined** | **442** | **34,090** | — | — | **50** | **242** |
| Detected AI subset | 206 | 17,121 | 83.11 | 80 | 50 | 215 |
| Missed AI subset | 15 | 1,446 | 96.40 | 91 | 58 | 140 |

AI answers were **13.77 words longer on average**. Despite the 80–120 word
instruction, **108/221 (48.87%)** AI answers were below 80 words, **100/221
(45.25%)** were within range, and **13/221 (5.88%)** exceeded 120. Length
instructions alone did not match the distributions.

| Set | Human mean / median words | AI mean / median words |
|---|---:|---:|
| 1 | 68.24 / 65 | 89.56 / 83.5 |
| 2 | 70.88 / 67.5 | 76.62 / 75 |
| 5 | 71.44 / 62 | 98.54 / 96 |
| 6 | 68.52 / 62 | 79.40 / 78.5 |
| 10 | 74.71 / 66 | 64.81 / 62 |

| Actual answer length | Human n | Human false positives | AI n | AI detected | AI missed | AI recall |
|---|---:|---:|---:|---:|---:|---:|
| 50–79 words | 169 | 0 | 108 | 102 | 6 | 94.44% |
| 80–120 words | 49 | 0 | 100 | 94 | 6 | 94.00% |
| 121+ words | 3 | 0 | 13 | 10 | 3 | 76.92% |

Missed AI answers were longer on average than detected AI answers, so shortness
alone does not explain the misses. These bins are exploratory: they have
unequal source counts, mix questions, and contain few long answers. They do not
constitute a controlled length-matched evaluation or establish a causal effect
of length.

## 7. Interpretation and next experiment

For this constrained baseline, the detector separates most fresh model answers
from the selected student responses. Performance varies by question; the pooled
score conceals substantially weaker detection on protein synthesis. The raw
polished class accounts for one-third of the website-verdict misses.

The result supplies a baseline against which prompt modifications can be tested.
There is **6.79 percentage points** of pooled AI-recall headroom under the primary
verdict, with more headroom in set 5. A sensible next experiment compares
original prompts, ordinary paraphrases, and Gemini-proposed task-preserving
modifications using the same model checkpoint, length conditions, retry policy,
and detector. Candidate selection and evaluation must use fresh independent
generation samples. Avoid tailoring all edits to these 15 missed answers and
then presenting the same answers as independent validation.

Human responses to the modified prompts are still needed: historical humans
answering original questions cannot establish that a modification increases
matched human–LLM divergence or preserves the human false-positive rate.

### Limits on claims

1. **Five prompts, one model.** No unseen-question, model-family, or CS transfer
   evaluation was performed.
2. **Length-conditioned humans.** Only longer student responses were eligible;
   the result does not cover the many short answers in SAS.
3. **Length-conditioned AI.** The extra instruction and retries change the
   model response distribution; all rejection rates must accompany results.
4. **Historical versus generated settings.** Students and the LLM did not share
   a controlled testing environment or matched ability/correctness distribution.
5. **Public data contamination.** The questions or answers could have appeared
   in model or detector training; membership has not been tested.
6. **Unverified answer correctness.** Detector performance is not evidence of
   educational quality, learning, or resistance to AI assistance.
7. **Time-based truncation of sampling.** Set 10 has only 21 samples per source;
   sequential generation and the timeout were not a randomized stopping design.
8. **Service calibration.** SnapJudge's guide describes calibration on research
   papers, not these student short answers. No local threshold calibration was done.
9. **No prompt-optimization result yet.** No divergence gain, causal stylistic
   mechanism, or advantage over random paraphrases has been demonstrated.

## 8. Reproducibility and provenance

All tables above were computed from the supplied archive rather than transcribed
from the chat. The analysis verifies source hashes; joins answers by opaque file
name; checks submitted document text against original answer IDs; verifies CSV
scores against raw JSON; and checks the no-save metadata and unique result names.
It reads the archive in place without extracting over project source files.

```bash
python3 experiments/analyze_snapjudge_baseline.py sas-report-data.tar.gz
```

This uses only Python's standard library and makes no detector calls. It writes
[the machine-readable metrics](snapjudge-sas-baseline.metrics.json), including
per-question counts, lengths, generation metadata, all missed-answer IDs and
source hashes. The supplied archive contains student text and is excluded from
Git; this report publishes aggregate results and generated-answer IDs only.

| Provenance item | Recorded identifier |
|---|---|
| Dataset revision | `f97ad6415c94c763b901423b72f78f1fbf46d630` |
| DeepSeek revision | `1df8507178afcc1bef68cd8c393f61a886323761` |
| Dataset-preparation Git commit | `296a0c6bccf3dd8edd403a90de06e5bde255ddc8` |
| Generation Git commit | `cede5d8692246f250732c832413a0a60b9c29327` |
| Reporting/archive Git commit | `b0d2ee4b5221e31da68f4b9b4fc8f06084237f8b` |
| SnapJudge client SHA-256 | `da8e3814a48af10a538a38a787901be542f519e121c61b7a3d8ab5994ae1f35f` |
| Python | 3.12.8 |
| torch / transformers | 2.6.0 / 4.51.3 |
| datasets / huggingface-hub | 3.5.0 / 0.30.2 |
| bitsandbytes / accelerate | 0.45.5 / 1.6.0 |
| tokenizers / CUDA runtime package | 0.21.4 / 12.4.127 |

The full package freeze is in the archive's `environment.txt`. Preparation,
generation, and reporting have different commits; a single final source snapshot
should not be mistaken for the exact historical code at every stage. This
experiment used SnapJudge, not Pangram; Pangram credit estimates are not
SnapJudge usage or a cost measurement for this run.

## Appendix: every AI answer missed by the website verdict

IDs use `question_id:zero_based_slot:attempt`. Each listed answer was a completed
fresh model output. “Polished” below is the detector's raw class, not its known
provenance. No human text is reproduced here.

| Set | Generated answer ID | Words | Raw class |
|---|---|---:|---|
| 1 | `f00fa49907fd1d05:4:1` | 75 | polished |
| 1 | `f00fa49907fd1d05:29:1` | 79 | polished |
| 2 | `a8967eb8986bb68d:13:1` | 58 | polished |
| 2 | `a8967eb8986bb68d:42:1` | 70 | polished |
| 5 | `d4e56e981111429a:2:1` | 111 | human |
| 5 | `d4e56e981111429a:9:1` | 121 | human |
| 5 | `d4e56e981111429a:12:1` | 115 | human |
| 5 | `d4e56e981111429a:19:1` | 115 | human |
| 5 | `d4e56e981111429a:22:1` | 75 | polished |
| 5 | `d4e56e981111429a:25:1` | 104 | human |
| 5 | `d4e56e981111429a:35:1` | 91 | human |
| 5 | `d4e56e981111429a:36:1` | 73 | human |
| 5 | `d4e56e981111429a:37:1` | 140 | human |
| 5 | `d4e56e981111429a:43:1` | 138 | human |
| 5 | `d4e56e981111429a:48:1` | 81 | human |

