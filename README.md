# Four-Criterion Evaluation of Emotion Representations

This research repository evaluates six linear emotion representations in
`allenai/Olmo-3-7B-Instruct` and `Qwen/Qwen2.5-14B-Instruct` using four criteria:

1. **Specificity** — does a vector distinguish its target emotion from neutral and other-emotion stories?
2. **Sensitivity** — does its readout increase across qualitative emotion-intensity levels?
3. **Invariance** — is the readout preserved under paraphrasing and translation?
4. **Functionality** — does intervening on the vector change open-ended emotional expression?

The evaluated emotions are happiness, surprise, fear, disgust, anger, and sadness.
Primary layers are Olmo L24 and Qwen L36, both at 75% relative depth.
The released vectors contain all 33 Olmo states and all 49 Qwen states;
L0 is the embedding and the last state is final-normalized.

The [English update notes](docs/BENCHMARK_UPDATE_NOTES_20261007.md) describe the
original problems, concrete examples and fixes under each of the four criteria.

## Corrected re-test (2026-10-04)

The vectors are unchanged. Statistics, materials, response scoring, language-quality
checks and the inference-to-metric workflow have been corrected and re-tested at L24.
See [re-test methods](docs/RETEST_20261004.md) and
[complete versioned results](results/retest_20261004/summary.json).

| Criterion | English | Chinese |
|---|---:|---:|
| Target–neutral AUROC | 0.910 | 0.852 |
| Six-way Top-1 | 0.55 | 0.49 |
| Intensity mean rho | 0.698 | 0.612 |
| Original–paraphrase rho | 0.863 | 0.905 |
| Anonymous functionality delta, ±0.5 | 0.347 | Quality failures prevent a clean overall claim |

Fixed-English-vector and fixed-Chinese-vector EN–ZH readout correlations are
0.810 and 0.777. At smaller doses, Chinese replacement-character rates drop
from 185/288 at ±0.5 to 32/288 at ±0.25 and 2/288 at ±0.1; baseline outputs have
none. Behavioral effects are smaller at the smaller doses. All raw failures remain.

## Qwen2.5-14B-Instruct evaluation (2026-10-04)

Qwen was evaluated with independently extracted bilingual vectors and the corrected
materials. All 5,000 readout rows and 2,076 newly generated, anonymously scored
answers are in [the Qwen results directory](results/qwen25_14b_20261004/).

| Metric | English | Chinese |
|---|---:|---:|
| Target–neutral AUROC | 0.867 | 0.856 |
| Target–other-emotion AUROC | 0.666 | 0.647 |
| Neutral-calibrated six-way Top-1 | 54.0% | 55.3% |
| Intensity mean rho | 0.646 | 0.614 |
| Original–paraphrase rho | 0.893 | 0.847 |

Ordinary English target-score delta at ±0.5 is 0.375, but **0/72 target pairs pass
the quality checks**. All four cross-language combinations at ±0.5 also have zero
quality-valid pairs. Smaller-dose coverage reaches at most 9/72. These results do
not establish stable, quality-preserving emotion control.

Five Chinese inputs are too short for token50 pooling and remain explicitly missing:
sensitivity uses 120 English / 117 Chinese ladders, invariance 596/600 complete pairs.
Qwen neutral PCA uses 858 paired texts, versus Olmo's 1,128. Native chat defaults,
generation configuration and intervention norms also differ between models.
See [Qwen methods and reproduction](docs/QWEN_20261004.md) and
[the full summary](results/qwen25_14b_20261004/summary.json).

## Historical Olmo results (2026-09-30)

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
