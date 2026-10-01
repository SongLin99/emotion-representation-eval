# Data card

All files are UTF-8 JSON Lines. English and Chinese texts remain paired in the same record.

| File | Rows | Purpose |
|---|---:|---|
| `vector_construction/emotion_stories.jsonl` | 6,000 | 1,000 stories for each of six emotions |
| `vector_construction/matched_neutral.jsonl` | 1,000 | 50 topics × 20 matched-neutral stories |
| `vector_construction/pca_neutral.jsonl` | 1,128 | Independent neutral material for PCA denoising |
| `specificity/stories.jsonl` | 350 | 50 events × six emotions plus neutral |
| `sensitivity/intensity_ladders.jsonl` | 600 | Six emotions × 20 events × five levels |
| `invariance/paraphrase_pairs.jsonl` | 600 | Original and paraphrased bilingual intensity stories |
| `functionality/prompts.jsonl` | 12 | Paired English/Chinese neutral open-ended prompts |

The vector-construction files contain the corrected English text actually used for the released
vectors and its paired Chinese translation. Five PCA texts shorter than the required pooling
boundary in the paired Llama Chinese pipeline were excluded in advance, leaving
1,128 PCA texts from an older 1,133-text neutral collection. This is independent
of the 50 × 20 matched-neutral set. The release retains that shared exclusion
policy for Olmo, rather than re-filtering the PCA collection using Olmo tokens.
Some `source_sha256` fields identify the original source English text, before
minimal translation-stage corrections; they are not hashes of the released text.
`MANIFEST.sha256` verifies the actual released files.

Specificity and sensitivity materials are independent of the vector-construction stories.
Invariance reuses the sensitivity items through paired meaning-preserving paraphrases.

## Known limitations

- Materials were generated and audited with model assistance, not fully human-authored or human-validated.
- Related negative emotions share appraisal structure, so cross-emotion confusion is expected.
- Chinese text readouts should be interpreted cautiously because Olmo-3-7B-Instruct is primarily an English checkpoint.
- Some narratives may mention loss, threat, contamination, or interpersonal conflict.
