# Known limitations of this frozen release

- **Automated, non-blind scoring.** Historical judge IDs disclosed the target
  emotion and intervention sign. Aggregate functionality scores are exploratory,
  not strictly blinded validation. The new anonymous scorer has not been run.
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
