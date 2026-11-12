# SnapJudge detection after exploratory Gemini rewrites

Recorded November 12, 2026. These aggregate results come from the cluster
`report.json` and Slurm output pasted by the user. The new per-answer results
and raw detector responses have not yet been transferred locally or audited.

## Question and result

Can Gemini rewrite educational questions so SnapJudge detects more DeepSeek
answers? In this run, detection rose from **206/221 (93.2%)** to
**214/221 (96.8%)**, an increase of **8 detections / 3.62 percentage points**.
This is a descriptive exploratory result, not a significance claim.

| Measure | Original questions | Exploratory rewrites |
|---|---:|---:|
| Unique questions | 5 | 5 |
| Eligible AI answers scored | 221 | 221 |
| Classified AI-generated | 206 | 214 |
| Classified Human-written | 15 | 7 |
| AI detection rate | 93.21% | 96.83% |
| Miss rate | 6.79% | 3.17% |
| Human answers scored in this run | 221 | 0 |

The original human baseline had zero false positives out of 221. No human
answers were collected for the rewritten questions, so there is no new
false-positive rate or complete rewritten-prompt accuracy estimate.

## Protocol and execution

- One Gemini 3.1 Pro rewrite per question, one shot, without detector feedback.
- The evaluated condition uses the **full saved Gemini outputs**, including
  rewritten background for sets 1 and 2. It does not use the versions displayed
  with the original background reattached.
- Exact rewrites: [frozen inputs](inputs/gemini-exploratory-v1.jsonl).
- DeepSeek-R1-Distill-Qwen-14B, pinned revision
  `1df8507178afcc1bef68cd8c393f61a886323761`, using the baseline's NF4
  quantization, sampling settings, seed scheme and answer-length instruction.
- Quotas: 50 accepted answers each for sets 1, 2, 5 and 6; 21 for set 10.
- Final answers must be complete and contain at least 50 words. Each answer
  slot has at most five attempts.
- Compatible saved attempts from the interrupted smaller run were reused.
- Output directory: `runs/sas-exploratory-221-v1/` on the cluster.

| Job | State | Exit code | Elapsed |
|---|---|---|---|
| Generation 13672811 | COMPLETED | 0:0 | 01:47:58 |
| Detection 13672840 | COMPLETED | 0:0 | 00:00:15 |

The eight-hour allocation was a maximum, not a required generation duration.
There were 223 saved attempts in the new report: 221 eligible answers and two
excluded answers under 50 words. All 221 eligible answers were scored, with
zero detector errors and zero missing results.

## Continuous scores

The new mean generated-window share is **0.9683257918552036**. This measures
the fraction of windows labeled generated; it is not authorship probability.

All 221 eligible answers have continuous scores. Their mean derived generated
decision margin is **3.3422080271493217**, compared with the previously extracted
baseline mean of approximately **3.0215**. The apparent increase is about
**0.321 raw score units**. This comparison is provisional until the new raw
responses confirm matching detector score definitions, versions and thresholds.

The derived margin is the minimum of the nonhuman threshold margin and the
generated-versus-polished threshold margin, averaged across windows. It is an
uncalibrated confidence proxy, not a probability; its average does not describe
the spread or uncertainty of individual scores.

## Interpretation and remaining checks

The aggregate detection rate increased with the same answer counts per question.
That supports further investigation of this particular rewrite condition.
However, only five questions were studied. Meaning preservation, changes in
reading difficulty, and answer correctness remain unverified. The model mainly
introduced technical language and sometimes additional demands, so this is not
yet evidence for a meaning-preserving intervention.

The comparison uses a historical baseline. Per-question detection and margin
changes, length distributions, detector consistency and uncertainty have not yet
been audited for the new run. Aggregate improvement could be concentrated in
one question. Answers are repeated samples within questions; they should not
be treated as 221 independent question-level replications.

No additional generation or detector submission is required to perform those
checks: the cluster's saved `questions.jsonl`, `answers.jsonl`, generation
manifest, and `snapjudge/` outputs contain the necessary data.

See the [original baseline report](001-snapjudge-sas-baseline.md) and
[rewrite protocol](003-exploratory-question-rewrites.md).
