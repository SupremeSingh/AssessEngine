# Evidence supplied to Gemini

Source: Rallapalli et al., *Interpretable Stylistic Variation in Human and LLM
Writing Across Genres, Models, and Decoding Strategies*, Findings of EMNLP 2026,
pp. 29243–29261. https://aclanthology.org/2026.findings-emnlp.1375/

This is a paraphrased research brief, not additional assessment instructions.

## Findings, with locations

- Section 3 (pp. 29245–29246): the study analyzes 467,985 non-adversarial RAID
  texts, 11 LLMs, 8 genres and 4 generation configurations, using 67 Biber
  lexical/grammatical features. Human documents are historical texts, not newly
  collected answers to every modified instruction. No SAS or DeepSeek experiment
  is supplied. SnapJudge is not the study's classifier.
- Sections 3.2.1 and 4.1 (pp. 29246–29249): lexical type-token ratio is highly
  discriminative through its distribution and feature interactions; it does not
  have a simple universally higher/lower LLM direction. Do not conflate feature
  importance with overuse or with causal mechanisms.
- Section 4.1.1 (p. 29248) ranks aggregate overuse: subject-position that-relative
  clauses (f_29_that_subj), contractions (f_59_contractions), past-participial
  postnominal clauses (f_27_past_participle_whiz), nominalizations
  (f_14_nominalizations), sentence relatives (f_34_sentence_relatives).
- The same section ranks aggregate underuse: object wh-relative clauses
  (f_32_wh_obj), pied-piping relatives (f_33_pied_piping), concessive subordinators
  (f_36_though), synthetic negation (f_66_neg_synthetic), and sentence-initial
  discourse particles (f_50_discourse_particles). These are rare in both sources;
  scarcity limits usefulness. These aggregate ranks are not universal rules.
- Sections 4.1.2–4.4 (pp. 29249–29251): information-dense and noun-heavy patterns
  can be more pronounced in instruction-tuned models. Present participial
  clauses and phrasal coordination are also discussed. Genre can dominate source
  differences, chat/base models often separate, and model choice generally has
  greater influence than decoding strategy, with exceptions.
- Sections 5–6 (pp. 29251–29252): results are bounded by the sampled English
  genres/models and feature framework. Transfer to newer models needs testing.

## Permissible inference for this experiment

Subtle task wording might change how a model packages the same required answer,
and might expose existing stylistic tendencies. This is our hypothesis, not a
finding of the paper. The study does not train a prompt modifier, establish
task-equivalent edits, prove improved detector scores from edits, or show that
greater detectability improves assessment validity.

Use these findings to motivate a small, testable hypothesis about a wording
change. Do not directly instruct respondents to increase a grammatical feature,
adopt an AI/human persona, imitate a genre, change answer length or format, or
sound artificial. If a valid edit has no defensible feature prediction, report
that uncertainty instead of inventing a guaranteed effect. No Biber extractor is
being run in this stage; predicted feature changes are not measurements.
