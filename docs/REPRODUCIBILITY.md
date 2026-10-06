# Reproducibility notes

## Low-cost verification

The repository contains frozen item-level readouts, responses, judgments, and summary files.
Run:

```bash
python scripts/verify_release.py
python scripts/summarize_results.py
python scripts/verify_updated_benchmarks.py
python -m unittest discover -s tests -v
```

This does not require a GPU or API key.

## Model inference

Vector construction is intentionally out of scope. Optional evaluation runners use
the released vectors and `allenai/Olmo-3-7B-Instruct`, pinned to revision
`096bb5469fe34348bc88d851a69edb3bf6f40df4`. Model weights are not included.

```bash
pip install -r requirements-inference.txt
python -m evaluation.run_readouts --task specificity --language en --output outputs/specificity_en.csv
python -m evaluation.run_readouts --task sensitivity --language zh --output outputs/sensitivity_zh.csv
python -m evaluation.run_readouts --task invariance --language en --output outputs/invariance_en.csv
python -m evaluation.run_functionality --output outputs/functionality.jsonl
python -m evaluation.run_functionality --cross-language --output outputs/cross_language.jsonl
```

Use a CUDA GPU with sufficient memory for the model and hidden states. Readouts
default to BF16 inference; `--dtype float32` is available for readout extraction.
New readout CSVs have a long-form schema. The scorer now converts schemas, pairs
invariance items by ID, and rejects missing pairs, nonfinite scores and mixed protocols:

```bash
python -m evaluation.score_readouts --task specificity --input outputs/specificity_en.csv --output outputs/specificity_metrics.json
python -m evaluation.score_readouts --task sensitivity --input outputs/sensitivity_zh.csv --output outputs/sensitivity_metrics.json
python -m evaluation.score_readouts --task invariance --input outputs/invariance_en.csv outputs/invariance_zh.csv --output outputs/invariance_metrics.json
```

Invariance requires both languages, each with original and paraphrased rows.
The versioned L24 re-test used the model in BF16 and retains raw item-level results.
Small hardware/version numerical
differences and, for greedy generation, different answers are possible.

## Model-assisted generation and scoring

Optional **new**, anonymized scoring requires an API key and costs API credits:

```bash
export OPENROUTER_API_KEY=...
python -m evaluation.judge_responses --input outputs/functionality.jsonl --output outputs/functionality_scored_v2.jsonl
```

The model identifier is `deepseek/deepseek-v4-flash-0731`. Because hosted model
behavior can change, exact regeneration is not guaranteed; released frozen materials should be
used for result verification. Anonymous shuffled IDs in this new runner differ
from historical scoring, which exposed conditions. Never relabel old scores as
blind or mix new scores with the frozen archive. Runners refuse to overwrite outputs.

API credentials are read only from `OPENROUTER_API_KEY`; no credential file is included in this
repository.

## Chinese replication

Chinese-prompt generation was often degraded under repeated interventions. A future replication
should use a balanced bilingual instruction model and prespecify language-quality exclusion rules.

## Corrected L24 re-test (2026-10-04)

Frozen original data and judgments are not overwritten. `data/revisions/20261004`
contains semantic revisions selected before remeasurement; `results/retest_20261004`
contains fresh readouts, anonymous rejudgments, smaller-dose generation and summary.
All 33-state original vectors are unchanged; this new evaluation focuses on L24.

```bash
python -m evaluation.retest_inference readouts
python -m evaluation.retest_inference doses
python -m evaluation.retest_summary --output outputs/olmo_recomputed_summary.json
```

The last command uses supplied frozen anonymous judgments; generating different
judgments with the optional API scorer will not reproduce exact hosted-model scores.
Small doses are ±0.1 and ±0.25, using all 12 prompts and all four prompt/vector
language combinations. Nonintervened Chinese baselines have no replacement characters;
damage is much more common at ±0.5, so base-language ability alone is not a complete explanation.

## Qwen L36 evaluation

The complete metrics can be reproduced offline from the released item-level files:

```bash
python -m evaluation.qwen_summary --output outputs/qwen_recomputed_summary.json
python scripts/verify_updated_benchmarks.py
```

The Qwen command handles the five short Chinese inputs using the frozen complete-case
policy; it does not fill missing scores. The generic `score_readouts` CLI rejects
incomplete conditions, so use `qwen_summary` for the combined Qwen table.

The current GPU runners above are Olmo-specific. Qwen inference was completed with
a separate research-workspace driver; that driver and the extraction implementation
are not included here. Frozen Qwen responses retain rendered native chat, token IDs,
batch sizes and editing norms. See [Qwen methods](QWEN_20261004.md).
