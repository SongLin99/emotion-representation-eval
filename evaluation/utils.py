"""Offline metric helpers; no model, API, or original-project imports."""
from pathlib import Path
import csv
import json
import math
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
EMOTIONS = ('happiness', 'surprise', 'fear', 'disgust', 'anger', 'sadness')


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def ranks(values):
    """One-based average ranks, including ties."""
    if any(not math.isfinite(x) for x in values):
        raise ValueError('Ranks require finite values')
    ordered = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for index in ordered[start:end]:
            result[index] = rank
        start = end
    return result


def pearson(a, b):
    if len(a) != len(b) or len(a) < 2:
        raise ValueError('Correlation requires equal arrays with at least two items')
    if any(not math.isfinite(x) for x in (*a, *b)):
        raise ValueError('Correlation requires finite values')
    ma, mb = mean(a), mean(b)
    da, db = [x-ma for x in a], [x-mb for x in b]
    denominator = math.sqrt(sum(x*x for x in da) * sum(x*x for x in db))
    # Historical convention: a constant readout receives zero rho.
    return sum(x*y for x, y in zip(da, db)) / denominator if denominator else 0.0


def spearman(a, b):
    return pearson(ranks(a), ranks(b))


def auc(positive, negative):
    """Probability a target exceeds a control, half credit for ties."""
    if not positive or not negative:
        raise ValueError('AUROC requires both classes')
    return mean(1.0 if p > n else 0.5 if p == n else 0.0
                for p in positive for n in negative)


def ccc(a, b):
    ma, mb = mean(a), mean(b)
    va, vb = mean((x-ma)**2 for x in a), mean((x-mb)**2 for x in b)
    denominator = va + vb + (ma-mb)**2
    covariance = mean((x-ma)*(y-mb) for x, y in zip(a, b))
    return 2*covariance / denominator if denominator else 0.0


def show(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def fractional_top1(values, target, tolerance=1e-12):
    """Equal credit among numerically tied maxima, independent of vector order."""
    maximum = max(values)
    winners = [i for i,x in enumerate(values) if abs(x-maximum) <= tolerance]
    return (1/len(winners) if target in winners else 0.0), len(winners)
