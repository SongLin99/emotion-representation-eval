# Four-Criterion Evaluation of Emotion Representations

This research repository evaluates six linear emotion representations in
`allenai/Olmo-3-7B-Instruct` and `Qwen/Qwen2.5-14B-Instruct` using four criteria:

1. **Specificity** — does a vector distinguish its target emotion from neutral and other-emotion stories?
2. **Sensitivity** — does its readout increase across qualitative emotion-intensity levels?
3. **Invariance** — is the readout preserved under paraphrasing and translation?
4. **Functionality (steering)** — does intervening on the vector change open-ended emotional expression?

The evaluated emotions are happiness, surprise, fear, disgust, anger, and sadness.
Primary layers are Olmo L24 and Qwen L36, both at 75% relative depth.
The released vectors contain all 33 Olmo states and all 49 Qwen states;
L0 is the embedding and the last state is final-normalized.

The [English update notes](docs/BENCHMARK_UPDATE_NOTES_20261007.md) describe the
original problems, concrete examples and fixes under each of the four criteria.

## Results: Olmo and Qwen (2026-10-04)

Both models use the corrected benchmark materials. Olmo was re-tested with its
existing vectors; Qwen uses independently extracted English and Chinese vectors.

| Metric | Olmo · English | Olmo · Chinese | Qwen · English | Qwen · Chinese |
|---|---:|---:|---:|---:|
| Emotion vs neutral (AUROC) | 0.910 | 0.852 | 0.867 | 0.856 |
| Emotion vs other emotions (AUROC) | 0.707 | 0.636 | 0.666 | 0.647 |
| Emotion classification accuracy | 55.0% | 49.0% | 54.0% | 55.3% |
| Intensity correlation (ρ) | 0.698 | 0.612 | 0.646 | 0.614 |
| Ladders increasing at every step | 21.7% | 16.7% | 18.3% | 19.7% |
| Paraphrase correlation (ρ) | 0.863 | 0.905 | 0.893 | 0.847 |
| Steering effect (±0.5, 180-token run) | 0.347 | — | 0.375 | — |
| Answer pairs passing quality checks (180-token run) | 72/72 | — | 0/72 | — |
| Steering effect (±0.5, 320-token run) | 0.278 | 0.333 | 0.153 | 0.764 |
| Answer pairs passing quality checks (320-token run) | 70/72 | 0/72 | 0/72 | 0/72 |

ρ is Spearman correlation; readout results are averaged across the six emotions.
Classification accuracy uses neutral stories from the other 49 events to calibrate
vector scores before predicting the emotion.

**Steering effect** is the average target-emotion score at +0.5 minus the score
at −0.5, on a 0–4 rating scale. Prompt and vector languages match in these rows.
The English-only run allows 180 new tokens; the bilingual run allows 320. A dash
means that condition was not run. Each pair contains a positive and a negative
intervention answer; the quality row counts pairs where both answers pass. **Length and reaching the
token cap alone are allowed**; 180/320 identifies the original generation run,
not a quality cutoff. Corrupted text, obvious repetition loops and poor semantic,
task or language quality still fail. Existing semantic ratings are retained.
Positive steering scores with few or no passing pairs need to be read alongside
the actual answers and quality checks.

Olmo Chinese replacement-character counts fall from 185/288 at ±0.5 to 32/288
at ±0.25 and 2/288 at ±0.1, with weaker steering effects at smaller doses.
Qwen has no target answer pairs passing quality checks at ±0.5; smaller-dose
coverage reaches at most 9/72. All failed answers remain in the results.

Qwen has five Chinese inputs too short for token50 pooling: intensity analysis uses
120 English / 117 Chinese ladders, and paraphrase/translation analysis uses 596/600
complete pairs. Qwen neutral PCA uses 858 paired texts, versus Olmo's 1,128.
Chat defaults, generation settings and intervention norms also differ.

- **Olmo:** [methods](docs/RETEST_20261004.md) · [complete results](results/quality_revision_20261007/olmo3_7b/summary.json)
- **Qwen:** [methods](docs/QWEN_20261004.md) · [complete results](results/quality_revision_20261007/qwen25_14b/summary.json)

[Quality-rule update and reproduction](docs/QUALITY_20261007.md). The original
2026-10-04 summaries remain available in their existing folders.

## Historical Olmo results (2026-09-30)

The original Olmo measurements and scores remain in the historical result folders.
For example, the original English steering effect was 0.500; the corrected score
is 0.347 for the same 180-token answers. Scoring anonymity and quality rules changed,
so the difference cannot be attributed to anonymization alone.

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
- The Qwen evaluation above extends the benchmark to another checkpoint; its quality failures remain visible too.

The Chinese results are retained rather than hidden because the failure pattern is itself
important for interpreting invariance and functionality.
The re-test shows a substantial intervention-dose contribution: this failure should
not be attributed exclusively to base Chinese-language ability.

## Repository contents

```text
data/          Frozen bilingual stories and evaluation materials
vectors/       English/Chinese vectors: 33 Olmo states and 49 Qwen states
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
python scripts/verify_updated_benchmarks.py
python -m unittest discover -s tests -v
```

`summarize_results.py` checks the original Olmo archive. `verify_updated_benchmarks.py`
recomputes both complete corrected Olmo and Qwen summaries, including bootstrap,
coverage and all dose/quality conditions, and compares them with the frozen files.
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
packaging. The original archives remain; the versioned corrected results are from
the completed 2026-10-04 inference and scoring runs.

## AI-assisted components

DeepSeek-V4-Flash-0731 was used to generate, translate, paraphrase, audit, or score some
materials. See [AI_USAGE.md](AI_USAGE.md). These model-assisted judgments are not human
validation.

## Status

This public repository is for research-group inspection and reproducibility. No
license has been assigned; public visibility does not grant a software/data license.
