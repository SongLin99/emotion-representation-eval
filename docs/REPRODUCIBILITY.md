# Reproducibility notes

## Low-cost verification

The repository contains frozen item-level readouts, responses, judgments, and summary files.
Run:

```bash
python scripts/verify_release.py
python scripts/summarize_results.py
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
New readout CSVs have a long-form schema and are not drop-in replacements for the
archived wide-format metric inputs. Invariance runs record both original and
paraphrased text. The portable runners have CPU unit tests, but were not fully
rerun on the model during release preparation. Small hardware/version numerical
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
