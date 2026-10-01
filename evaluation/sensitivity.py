"""Rank correlation inside each five-level ladder."""
from collections import defaultdict
from statistics import mean, median
from .utils import ROOT, EMOTIONS, read_csv, show, spearman


def analyze(rows=None):
    rows = read_csv(ROOT/'results/sensitivity/item_scores_l24.csv') if rows is None else rows
    result = {}
    for language in ('en', 'zh'):
        groups = defaultdict(list)
        for row in rows:
            if row['language'] == language:
                groups[row['topic_id']].append(row)
        records = []
        for topic, group in sorted(groups.items()):
            group.sort(key=lambda r: int(r['level']))
            if [int(r['level']) for r in group] != list(range(5)):
                raise ValueError('Incomplete ladder: '+topic)
            emotion = group[0]['emotion']
            values = [float(r['mean_'+emotion]) for r in group]
            tail = [float(r['tail16_'+emotion]) for r in group]
            records.append(dict(topic_id=topic, emotion=emotion,
                rho=spearman(list(range(5)), values), tail16_rho=spearman(list(range(5)), tail),
                adjacent_accuracy=mean(b > a for a, b in zip(values, values[1:])),
                strict_monotonic=all(b > a for a, b in zip(values, values[1:])),
                high_low_delta=values[-1]-values[0]))
        def summarize(items):
            return dict(n_ladders=len(items), mean_rho=mean(r['rho'] for r in items),
                median_rho=median(r['rho'] for r in items),
                adjacent_accuracy=mean(r['adjacent_accuracy'] for r in items),
                strict_monotonic=mean(r['strict_monotonic'] for r in items),
                mean_high_low_delta=mean(r['high_low_delta'] for r in items),
                mean_tail16_rho=mean(r['tail16_rho'] for r in items))
        result[language] = dict(macro=summarize(records), per_emotion={e: summarize(
            [r for r in records if r['emotion'] == e]) for e in EMOTIONS})
    return result


if __name__ == '__main__':
    show(analyze())
