# Known limitations of this frozen release

The historical limitations below remain attached to the original archive.
The 2026-10-04 re-test fixes scoring anonymity, bootstrap event identity, slope-ranking
ties and fresh-inference schema integration, with revised materials in separate folders.
It does not retroactively turn old scores into blind judgments or human-validate the stories.

- **Automated, non-blind scoring.** Historical judge IDs disclosed the target
  emotion and intervention sign. Aggregate functionality scores are exploratory,
  not strictly blinded validation. The corrected Olmo and Qwen results use new
  anonymous shuffled scoring; the historical scores remain unchanged.
- **Chinese generation quality.** In the cross-language experiment, 185 of 300
  Chinese-prompt responses contain the Unicode replacement character. Some still
  received high automated language scores. Thus the large Chinese steering scores
  are not reliable evidence of clean functional transfer. All failures remain public.
- **Emotion specificity is limited.** Target–neutral separation is stronger than
  separation of nearby emotions. Neutral-calibrated Top-1 is 55% English / 49%
  Chinese; raw Top-1 is 51% / 47.3%. Neither is perfect construct identification.
- **Qualitative intensity is imperfect.** The archived model-assisted audit
  flags 7/120 ladders: happiness T04/T13, sadness T03, surprise T09/T12/T13/T14.
  All remain in the primary results. Only 21.7% / 15.0% of English / Chinese
  ladders are strictly increasing. Some neutral narratives are debatable;
  generated materials have not received exhaustive human validation.
- **Historical bootstrap caveat.** Specificity bootstrap code relabeled repeated
  sampled events as separate events; their duplicate neutral stories could enter
  the leave-one-event-out calibration. Archived calibrated-ranking confidence
  intervals should not be used as validated intervals. The non-bootstrap point
  estimates use the other 49 events and are independently reproduced here.
- **Language comparisons.** Invariance uses language-specific vectors and paired
  translations. High rank agreement is not proof of equal scales or a universal
  language-independent representation. Chinese generation failures motivate a
  replication with a stronger bilingual model, not deletion of negative results.
- **Scope.** Four properties of linear directions do not establish subjective
  emotion, consciousness, or introspective access. Vector-construction code and
  report pages are intentionally excluded from this evaluation-only repository.

## Qwen extension

Qwen L36 uses 858 paired neutral PCA texts instead of Olmo's 1,128, and five Chinese
benchmark inputs cannot support token50 readouts. The complete-case policy retains
117 Chinese sensitivity ladders and 596 invariance pairs. Model comparisons also
include native chat, generation settings and model-specific norms.

Qwen's positive raw ±0.5 score changes have no quality-valid target pairs. Small-dose
quality coverage is low too. Quality-filtered subsets can be selected and never
replace full-population effects. Neither a stronger checkpoint nor a positive score
delta automatically validates functionality.

Single-model judgments and model-assisted materials are not independent human
validation. The corrected rubric additionally penalizes corrupted text, so score
changes from the historical archive cannot be attributed only to anonymization.
