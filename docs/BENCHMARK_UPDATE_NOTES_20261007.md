# Benchmark update notes

October 7, 2026. This records the corrected Olmo L24 evaluation and the new Qwen2.5-14B-Instruct L36 evaluation.

## Specificity

The calibration bootstrap could leak held-out event information. If event A was sampled twice, excluding only one copy left the other in the neutral calibration set. A story also had a physical plausibility problem: a hot bowl taken straight from a kiln contained fermented residue and smelled moldy, implying that the contamination survived firing.

Calibration now excludes every sampled copy of the held-out original event. The pottery outcome was revised so contamination occurs after removal from the kiln, while keeping the shared setup. Continuation-only readouts were added to check the influence of the common story background.

## Sensitivity

Some ladders started with an outcome that was already emotional. In `anger_T15`, the lowest level described a well-written essay receiving a low grade without feedback while other students received comments. That already suggests unfair treatment. The comparison of which vector best tracks intensity also favored earlier emotions when several vectors tied for the highest correlation.

The neutral outcome now follows the published grading rules, with unfairness increasing at later levels. Nine ladders, or 45 stories, were revised before remeasurement; all 120 Olmo ladders remain. Tied maxima receive equal credit: two tied vectors each get 0.5.

Three Qwen Chinese originals have fewer than 50 tokens. Their texts remain unchanged and their readouts are explicitly missing. Chinese sensitivity uses 117 complete ladders, with an additional comparison on the same 117 ladders in both languages.

## Invariance

Overall paraphrase correlation can be inflated by shared intensity ordering: low-intensity originals and paraphrases both score low, and high-intensity items both score high. This does not establish agreement among stories at the same level. Using English vectors for English and Chinese vectors for Chinese also does not test whether one direction works across languages.

The update adds within-level paraphrase correlations and cross-language comparisons with the vector language held fixed. Paraphrases for the 45 revised stories were updated, and original/paraphrase/language conditions are checked by ID. Qwen uses 596 of 600 complete four-condition pairs; exclusion records remain available. The local report shows actual paired readouts.

## Functionality (steering)

Historical IDs exposed the intervention, such as `desk_target_happiness_-0.5`, which could influence the judge. Emotion-score changes also occurred alongside damaged responses: 185 of 288 Olmo Chinese intervention answers at ±0.5 contained replacement characters.

The judge now receives shuffled anonymous IDs, prompts and answers, without intervention assignments. The 900 historical Olmo answers were re-scored, and 1,176 new answers were generated at ±0.1 and ±0.25. Corruption, repetition, language, semantic coherence and task adherence are checked.
Length and token-cap hits are now recorded without automatically failing an answer. Full effects, steering effects for answers passing quality checks and coverage are reported together, with failures retained.

At the smaller doses, Olmo Chinese replacement-character counts fall to 32/288 and 2/288, but emotion effects weaken too. Qwen uses a separately calibrated intervention norm and has 2,076 newly generated and scored answers. Its target interventions at ±0.5 have no answer pairs passing quality checks; smaller-dose coverage reaches at most 9/72. The evaluation and quality checks have improved, but stable emotion control remains unresolved.

The results display was also updated on October 7: metric names now describe what
they measure, including “Steering effect”, and one comparison table shows both
models and languages. The 180-token and 320-token steering runs are labeled
separately. Calculations and frozen results are unchanged.

The quality rule was revised at the user’s request on October 7: long coherent
answers are allowed. Reaching the generation cap no longer causes rejection;
corruption, repetition and poor semantic quality still do. The updated summaries
are stored separately, with original answers and ratings retained. See
[quality update](QUALITY_20261007.md).

Details: [corrected Olmo re-test](RETEST_20261004.md), [Qwen evaluation](QWEN_20261004.md).
