# Methods summary

## Representation readout

For a story, the hidden states from token 50 onward are averaged. The representation readout is
the cosine similarity between this pooled state and the same-language unit emotion vector.
Stories are raw text, with no chat template or rating question. The 50th token is
included, counted among valid tokens separately for each story. Primary results use L24.

## Specificity

Fifty events each have seven matched continuations: six emotional outcomes and one neutral
outcome. Metrics include emotion vs neutral AUROC, emotion vs other emotions AUROC, and calibrated
classification accuracy across six emotions. AUROC is the fraction of target/comparison pairs with a
higher target readout (ties count as half). For ranking, each vector's readout is
standardized using neutral stories from the other 49 events; the largest of the
six standardized readouts predicts the label. Classification accuracy without
neutral calibration is also released.

## Sensitivity

Each emotion has 20 independent events and five qualitative outcome levels per event. No numeric
X value, rating, explicit intensity adjective, or described emotional reaction appears. Spearman
rho is computed between level indices 0–4 and the five target-vector cosine
readouts within each ladder, then averaged equally across 120 events. Adjacent
increase accuracy and the fraction of strictly increasing ladders are also reported.
Levels specify intended qualitative order, not validated equal intensity intervals.
For the auxiliary comparison of six slope correlations, tied maxima now receive
shared credit for the highest intensity correlation (tolerance 1e-12); this does not change mean target rho.

## Invariance

The 600 sensitivity stories receive meaning-preserving paraphrases in both languages.
Spearman correlation measures preservation of ordering; score agreement (CCC)
also checks whether paired readings have the same scale. Readouts are paired item by item within each emotion
(100 items per emotion), correlations are computed separately, and the six
emotion results are averaged. English–Chinese comparisons use the respective
language-specific vectors; they do not establish a single language-independent vector.
The corrected re-test additionally holds the vector language fixed for paired
English/Chinese comparisons and reports within-level paraphrase correlations.

## Functionality (steering)

At L24, an emotion vector scaled by alpha = -0.5 or +0.5 is added at the assistant boundary and
each later decoding position. Equal-norm directions orthogonal to the span of all six emotion
vectors are controls. The edit is h' = h + alpha * N * v, with a unit vector v and
fixed neutral-token norm N = 35.4885577. Only the final assistant boundary in
prefill and each subsequently generated assistant token are edited; user-token
states are not edited. Greedy generation is used. A model-assisted judge rates
six emotions from 0 to 4. Historical scoring IDs revealed the conditions, so the
judge was not strictly blinded. The primary contrast is the target-emotion score
at +0.5 minus the score at -0.5, averaged across matched prompts/emotions.
The ordinary experiment includes 300 responses (12 prompts, one baseline and
24 interventions each); the cross-language experiment includes 600 responses
(both prompt languages, two vector languages, one baseline per prompt language).
The corrected re-test anonymously re-scores those 900 responses and adds 1,176
responses at ±0.1 and ±0.25 (24 baselines plus 1,152 interventions). Objective
replacement-character, truncation and requested-script checks supplement judge quality.
Steering effects and pair counts are reported for all answers and for answer pairs
passing quality checks.

## Qwen extension

Qwen2.5-14B-Instruct uses the same corrected material definitions and metric code,
with an independently fixed L36 primary state and model-specific norm
N=355.8481024546985. Its vectors contain 49 states. Native Qwen chat/generation
defaults differ from Olmo; five short Chinese token50 inputs are explicitly missing,
and PCA uses 858 paired neutral texts. Coverage and complete-case selection are
documented in [Qwen methods](QWEN_20261004.md).

## Interpretation boundary

These tests evaluate properties of linear emotion-related directions. They do not establish
subjective emotion or conscious experience.
