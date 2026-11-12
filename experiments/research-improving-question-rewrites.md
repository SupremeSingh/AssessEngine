# Research note: improving Gemini's question rewrites

**Date:** November 11, 2026. **Scope:** research recommendations, not a new
generation run or a claim that detection improved. No Gemini/detector calls
were made for this research step. The one-rewrite-per-question design remains.

## Main finding

More elaborate question prose is not a reliable optimization objective. The
strongest next improvement is to give the modifier a precise task contract and,
after a development experiment, measured feedback about answering-model behavior.
The stylistic papers provide hypotheses; they do not establish a recipe for
increasing SnapJudge scores on task-equivalent SAS questions.

Our earlier description of small rewrites as inherently weak was too strong.
They may be unconvincing as causal explanations without being ineffective.
Only target-model measurements can resolve that distinction.

## Evidence and what transfers

| Primary source | What it establishes | Application here | Limit of the inference |
|---|---|---|---|
| [Wahle et al., EMNLP 2024: Paraphrase Types Elicit Prompt Engineering Capabilities](https://aclanthology.org/2024.emnlp-main.617/) | Evaluates six paraphrase families across five models and 120 tasks; effects vary and lexical/morphological changes can matter | Label the linguistic edit rather than require a visually dramatic rewrite | Task-performance gains are not evidence of detection gains |
| [Elgaar and Amiri, Findings 2025: Linguistically-Controlled Paraphrase Generation](https://aclanthology.org/2025.findings-emnlp.1137/) | Separates semantic fidelity from control of linguistic attributes, using a specialized paraphrase model and quality-control mechanism | Specify fixed content separately from a chosen linguistic transformation | Does not show that a natural-language Gemini prompt reproduces LingConv's control |
| [Reinhart et al., PNAS 2025: Do LLMs Write Like Humans?](https://doi.org/10.1073/pnas.2422455122) | Finds grammatical/style differences in parallel human/LLM corpora, including noun-heavy preferences in instruction-tuned models | Keep nominalization, participial clauses and coordination as candidate answer-level hypotheses | Model/genre effects and observed feature preferences do not prove a prompt intervention |
| [Pryzant et al., EMNLP 2023: ProTeGi](https://aclanthology.org/2023.emnlp-main.494/) | Uses natural-language critiques of observed errors to direct prompt editing | Give Gemini specific evidence about why its current rewrite did not affect responses | Our simple one-rewrite loop is an adaptation, not a reproduction of its beam-search algorithm |
| [Yang et al., ICLR 2024: OPRO](https://arxiv.org/abs/2309.03409) | Uses previous proposals and measured objective values as optimization context | Supply previous wording and development-batch score changes to guide one revision | Reported benchmarks optimize different objectives; success on our detector is unproven |
| [Yadla, *SEM 2026: Syntactic Priming in Few-Shot Learning](https://aclanthology.org/2026.starsem-conference.2/) | Finds construction-specific priming from syntactic demonstrations in several model families/tasks | Treat response mirroring as a testable mechanism | Evidence concerns few-shot demonstrations and different constructions, not our single-question relative-clause edits |

These sources motivate experiments, not a claim that humans are unaffected by
the same wording changes. No reviewed source directly establishes our desired
combination: meaning-preserving educational edits, DeepSeek 14B, and higher
SnapJudge scores with unchanged human false positives.

## Five additions to the modifier instructions

### 1. Define a task contract before editing

Have Gemini extract: knowledge assessed, required answer components, item counts,
quantifiers, uncertainty, starting point, supporting evidence, and response format.
Separate immutable source context from editable task phrasing. This is our
operational recommendation, not a task-equivalence theorem from these papers.

For example, the protein question requires four described steps, starting with
mRNA leaving the nucleus. The doghouse question requires an evidence-backed
choice and an account of how it *might* affect the interior. A style intervention
cannot silently turn those into a more detailed mechanism assessment or a claim
of certain real-world effects.

### 2. Choose one identifiable linguistic lever

Allow a lexical, syntactic, or information-organization change. Ask Gemini to
identify the lever and predict its effect on the answer. Do not force all edits
toward nominalization or assume increased complexity is desirable. The natural
language of the same task can change without adding an answer obligation.

Useful hypotheses include entity-centered versus action-centered phrasing,
shifting information focus, and regrouping already-required subparts. These are
candidate interventions, not proven ways to cause machine-like text. Retain
scientific meanings and the original response format.

### 3. Require a counterexample check

Ask: could a correct answer satisfy the original but fail the rewrite, or vice
versa, solely because the rewrite adds/removes a demand? Have Gemini attempt to
find such a counterexample before returning its one final edit. If found, revise
the edit or report that a valid intervention was not found.

This check is stronger than character similarity but is still model judgment.
It should not be presented as independent human validation or formal entailment.

### 4. Give measured development feedback

For each original and current rewrite, generate a small development batch under
the same DeepSeek checkpoint, settings, length instruction and retry rules. Feed
Gemini the wordings, a few representative generated answers, mean/dispersion of
the continuous margin, word counts, and rejection counts. Give no hidden test
results. Ask it to diagnose a specific observed failure and return one revised
question, rather than a pool of speculative candidates.

This follows the feedback principle of ProTeGi/OPRO without adopting their full
search procedures. For the first round, start from the available paper context;
there is no modified-answer feedback yet. Do not falsely tell Gemini the latest
rewrites improved or failed: they have not been tested.

### 5. Explicitly reject shortcuts in the optimization objective

The objective is higher *generated decision margin*, with task preservation and
comparable response conditions. It is not longer answers, more boilerplate,
answer leakage, changed correctness, a new writing genre, or explicit directions
to sound artificial. Word counts and retry rates are diagnostics, not secondary
targets to exploit. Character-level similarity alone is not task equivalence.

## Suggested instruction block

> Rewrite this assessment once, preserving the set of acceptable answers and
> the required knowledge, evidence, counts, scope, uncertainty and answer format.
> First establish those invariants. Then choose one linguistic or information-focus
> change that might alter how DeepSeek expresses the same answer. The research
> findings are suggestions, not requirements to insert particular grammar.
>
> Predict the answer-level change and name the relevant linguistic feature. A
> small edit is acceptable if its hypothesis is specific; a larger edit is not
> better simply because it looks different. Do not add reasoning demands or
> stylistic/length instructions to the respondent-facing question.
>
> Before returning your single rewrite, try to construct an answer that would
> be valid under one wording but invalid under the other. If the mismatch comes
> from changed requirements, repair the rewrite. Explain the remaining uncertainty.
>
> When development feedback is supplied, use the observed responses and score
> changes to diagnose the previous edit. Do not invent results. Optimize the
> continuous detector margin without exploiting answer length or changing the
> assessment task. Return one final rewrite and a concise rationale.

## Practical recommendation for this project

Do not make another unmeasured rewrite round the sole next step. Correct the
known uncertainty drift in set 10, inspect the task contract for each existing
proposal, and run a small original-versus-modified development batch. Then give
Gemini the resulting critique if the effect is weak or negative.

Ten qualifying answers per original/modified condition is 100 answers across five
questions, suitable for an exploratory signal, not a powered confirmatory claim.
If revising based on those scores, confirm on fresh independent generations.
Generation seeds held out within these five prompts are **not** held-out questions.
Use question-level reporting, same detector thresholds/version, and length/retry
diagnostics. A detector-score gain on AI alone is still not evidence of greater
human–LLM divergence; modified questions eventually require human responses.
