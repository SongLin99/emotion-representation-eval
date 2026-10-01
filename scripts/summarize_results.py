from __future__ import annotations

import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import specificity, sensitivity, invariance, functionality


def read(relative: str):
    return json.loads((ROOT / relative).read_text())


def main() -> None:
    for name, (value, archived) in verified_metrics().items():
        if not math.isclose(value, archived, rel_tol=1e-7, abs_tol=1e-8):
            raise ValueError(f'{name}: recomputed {value} != archived {archived}')
        print(f"{name}: {value:.6f}")
    print('PASS: main metrics independently recomputed from released item-level data')
    print('Caution: functionality judgments are historical, automated, and not strictly blinded.')


def verified_metrics():
    sp, se, inv, fun = specificity.analyze(), sensitivity.analyze(), invariance.analyze(), functionality.analyze()
    old_sp = read('results/specificity/summary.json')['languages']
    old_se = read('results/sensitivity/summary.json')['languages']
    old_inv = read('results/invariance/summary.json')['primary']
    old_fun = read('results/functionality/results.json')['primary']
    old_cross = read('results/crosslingual_functionality/summary.json')['primary']
    values = {}
    for lang in ('en', 'zh'):
        for metric in ('target_neutral_macro', 'cross_emotion_macro', 'top1', 'top2', 'mrr'):
            values[f'specificity_{metric}_{lang}'] = (sp[lang][metric], old_sp[lang]['primary'][metric])
        for metric in ('mean_rho', 'adjacent_accuracy', 'strict_monotonic'):
            values[f'sensitivity_{metric}_{lang}'] = (se[lang]['macro'][metric], old_se[lang]['primary']['macro'][metric])
    for condition in inv:
        for metric in ('macro_target_spearman', 'macro_target_ccc'):
            values[f'invariance_{condition}_{metric}'] = (inv[condition][metric], old_inv[condition][metric])
    values['functionality_target_delta'] = (fun['target']['mean_delta'], old_fun['target_mean_delta'])
    values['functionality_target_minus_random'] = (fun['target_minus_random'], old_fun['target_minus_random'])
    values['cross_language_minus_same'] = (fun['cross_language']['cross_minus_same'], old_cross['paired']['all']['cross_minus_same'])
    return values


if __name__ == "__main__":
    main()
