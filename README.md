# Four-Criterion Evaluation of Emotion Representations in Olmo-3-7B-Instruct

This research repository evaluates six linear emotion representations in
`allenai/Olmo-3-7B-Instruct` using four criteria:

1. **Specificity** — does a vector distinguish its target emotion from neutral and other-emotion stories?
2. **Sensitivity** — does its readout increase across qualitative emotion-intensity levels?
3. **Invariance** — is the readout preserved under paraphrasing and translation?
4. **Functionality** — does intervening on the vector change open-ended emotional expression?

The evaluated emotions are happiness, surprise, fear, disgust, anger, and sadness.
The primary analysis layer is L24 (approximately 75% depth), while the released vector
files contain all 33 recorded states: L0 embedding, L1–31 block outputs, and L32 final
normalized state.

## Main results

| Criterion | English | Chinese | Interpretation |
|---|---:|---:|---|
| Specificity: target–neutral AUROC | 0.910 | 0.852 | Strong target–neutral separation |
| Specificity: neutral-calibrated six-way Top-1 | 0.55 | 0.49 | Moderate fine-grained emotion separation |
| Sensitivity: mean within-ladder Spearman rho | 0.697 | 0.599 | Readouts track qualitative intensity |
| Invariance: original–paraphrase rho | 0.859 | 0.905 | Strong within-language paraphrase stability |
| Invariance: English–Chinese rho | 0.769 | — | Ordering transfers better than absolute calibration |
| Functionality: target-vector delta | 0.500 | — | Aggregate behavioral effect; disgust has zero mean effect |

All null, negative, and low-quality conditions are retained in `results/`.
Historical functionality scoring was **not strictly blinded**: condition IDs exposed
the emotion and intervention sign. The scores are exploratory automated judgments,
not independent validation. See [limitations](docs/LIMITATIONS.md).

## Important multilingual limitation

Olmo-3-7B-Instruct is released primarily as an English-language checkpoint; it is not a
balanced bilingual English–Chinese model. Chinese story readouts are informative, but
repeated interventions during Chinese generation frequently produce malformed or
mixed-language text. Therefore:

- Chinese specificity, sensitivity, and invariance results are reported as preliminary cross-language evidence.
- Chinese-prompt steering is not treated as clean evidence of functional equivalence.
- The cleanest transfer result is a Chinese-derived vector steering an English response.
- A future replication should use a model explicitly trained and instruction-tuned for both English and Chinese.

The Chinese results are retained rather than hidden because the failure pattern is itself
important for interpreting invariance and functionality.

## Repository contents

```text
data/          Frozen bilingual stories and evaluation materials
vectors/       English and Chinese emotion vectors for all 33 recorded states
evaluation/    Standalone four-criterion metrics and optional inference/scoring runners
results/       Item-level readouts, generated answers, automated scores, and summaries
docs/          Data card, methods, and reproduction notes
prompts/       Evaluation prompt provenance
```

## Vector construction

For each language and layer:

1. Pool each story's hidden states from the 50th valid token onward.
2. For each of 50 shared topics, subtract the matched-neutral mean from the target-emotion mean.
3. Average the 50 topic differences equally.
4. Remove neutral-text principal components explaining at least 50% of variance.
5. Normalize the residual vector to unit length.

No emotion-rating question or role-play instruction is used during vector extraction.
This repository releases the precomputed vectors and their materials, not the
vector-extraction/PCA implementation. Interactive HTML reports are also excluded.

## Quick inspection

```bash
pip install -r requirements.txt
python scripts/verify_release.py
python scripts/summarize_results.py
python -m unittest discover -s tests -v
```

`summarize_results.py` recomputes the main L24 metrics from item-level data and checks
them against the archived summaries; it does not just print the saved summaries.
The metric modules use Python's standard library and can also be run separately:

```bash
python -m evaluation.specificity
python -m evaluation.sensitivity
python -m evaluation.invariance
python -m evaluation.functionality
```

`MANIFEST.sha256` records checksums for all released files except itself.

Optional GPU inference and anonymous rescoring are described in
[reproduction notes](docs/REPRODUCIBILITY.md). These portable runners were prepared
and unit-tested for this release; full GPU/API reruns were **not** performed during
packaging. The original data and scores were not regenerated or modified.

## AI-assisted components

DeepSeek-V4-Flash-0731 was used to generate, translate, paraphrase, audit, or score some
materials. See [AI_USAGE.md](AI_USAGE.md). These model-assisted judgments are not human
validation.

## Status

This public repository is for research-group inspection and reproducibility. No
license has been assigned; public visibility does not grant a software/data license.
