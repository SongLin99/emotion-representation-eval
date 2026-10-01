"""Target/neutral separation and held-out-topic neutral-calibrated ranking."""
from statistics import mean, stdev
from .utils import ROOT, EMOTIONS, auc, read_csv, show


def analyze(rows=None):
    rows = read_csv(ROOT/'results/specificity/item_scores_l24.csv') if rows is None else rows
    result = {}
    for language in ('en', 'zh'):
        subset = [r for r in rows if r['language'] == language]
        neutral = [r for r in subset if r['condition'] == 'neutral']
        target_auc, cross_auc = {}, {}
        for emotion in EMOTIONS:
            field = 'cos_' + emotion
            target = [float(r[field]) for r in subset if r['condition'] == emotion]
            target_auc[emotion] = auc(target, [float(r[field]) for r in neutral])
            cross_auc[emotion] = {other: auc(target, [float(r[field]) for r in subset
                                                  if r['condition'] == other])
                                  for other in EMOTIONS if other != emotion}
        correct_ranks, raw_hits = [], []
        for row in subset:
            if row['condition'] == 'neutral':
                continue
            calibration = [r for r in neutral if r['topic_id'] != row['topic_id']]
            z = []
            for emotion in EMOTIONS:
                field = 'cos_' + emotion
                values = [float(r[field]) for r in calibration]
                z.append((float(row[field])-mean(values))/stdev(values))
            target = EMOTIONS.index(row['condition'])
            order = sorted(range(6), key=lambda i: (-z[i], i))
            correct_ranks.append(order.index(target)+1)
            raw_hits.append(max(EMOTIONS, key=lambda e: float(row['cos_'+e])) == row['condition'])
        result[language] = dict(
            n=len(subset), target_neutral_auc=target_auc, cross_emotion_auc=cross_auc,
            target_neutral_macro=mean(target_auc.values()),
            cross_emotion_macro=mean(v for d in cross_auc.values() for v in d.values()),
            top1=mean(r == 1 for r in correct_ranks), top2=mean(r <= 2 for r in correct_ranks),
            mrr=mean(1/r for r in correct_ranks), uncalibrated_top1=mean(raw_hits))
    return result


if __name__ == '__main__':
    show(analyze())
