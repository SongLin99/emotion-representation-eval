"""Rank correlation inside each five-level ladder."""
from collections import defaultdict
from statistics import mean, median
from .utils import ROOT, EMOTIONS, read_csv, show, spearman, fractional_top1
from .schema import readout_rows


def analyze(rows=None):
    rows = read_csv(ROOT/'results/sensitivity/item_scores_l24.csv') if rows is None else rows
    rows = readout_rows(rows, 'sensitivity')
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
            if len({r['emotion'] for r in group}) != 1:
                raise ValueError('Mixed emotions within ladder: '+topic)
            values = [float(r['mean_'+emotion]) for r in group]
            all_rho = [spearman(list(range(5)), [float(r['mean_'+e]) for r in group]) for e in EMOTIONS]
            credit, n_tied = fractional_top1(all_rho, EMOTIONS.index(emotion))
            tail = [float(r['tail16_'+emotion]) for r in group] if all('tail16_'+emotion in r for r in group) else None
            records.append(dict(topic_id=topic, emotion=emotion,
                rho=spearman(list(range(5)), values), tail16_rho=spearman(list(range(5)), tail) if tail is not None else None,
                all_vector_rhos=all_rho, top1_credit=credit, n_tied_maxima=n_tied,
                adjacent_accuracy=mean(b > a for a, b in zip(values, values[1:])),
                strict_monotonic=all(b > a for a, b in zip(values, values[1:])),
                high_low_delta=values[-1]-values[0]))
        if not records:
            continue
        def summarize(items):
            return dict(n_ladders=len(items), mean_rho=mean(r['rho'] for r in items),
                median_rho=median(r['rho'] for r in items),
                adjacent_accuracy=mean(r['adjacent_accuracy'] for r in items),
                strict_monotonic=mean(r['strict_monotonic'] for r in items),
                mean_high_low_delta=mean(r['high_low_delta'] for r in items),
                fractional_slope_top1=mean(r['top1_credit'] for r in items),
                tied_maximum_fraction=mean(r['n_tied_maxima'] > 1 for r in items),
                mean_tail16_rho=mean(r['tail16_rho'] for r in items) if all(r['tail16_rho'] is not None for r in items) else None)
        result[language] = dict(macro=summarize(records), records=records, per_emotion={e: summarize(
            [r for r in records if r['emotion'] == e]) for e in EMOTIONS if any(r['emotion']==e for r in records)})
    return result


if __name__ == '__main__':
    show(analyze())
