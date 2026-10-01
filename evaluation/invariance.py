"""Paired readout ordering and concordance, averaged across emotions."""
from statistics import mean
from .utils import ROOT, EMOTIONS, ccc, pearson, read_csv, show, spearman


def analyze(rows=None):
    rows = read_csv(ROOT/'results/invariance/item_scores_l24.csv') if rows is None else rows
    result = {}
    for name, a, b in (
        ('english_paraphrase', 'original_en_target', 'paraphrase_en_target'),
        ('chinese_paraphrase', 'original_zh_target', 'paraphrase_zh_target'),
        ('cross_language_original', 'original_en_target', 'original_zh_target'),
        ('cross_language_paraphrase', 'paraphrase_en_target', 'paraphrase_zh_target')):
        per = {}
        for emotion in EMOTIONS:
            group = [r for r in rows if r['emotion'] == emotion]
            av, bv = [float(r[a]) for r in group], [float(r[b]) for r in group]
            per[emotion] = dict(n=len(group), spearman=spearman(av, bv),
                pearson=pearson(av, bv), ccc=ccc(av, bv),
                bias=mean(y-x for x, y in zip(av, bv)), mae=mean(abs(y-x) for x, y in zip(av, bv)))
        result[name] = dict(macro_target_spearman=mean(r['spearman'] for r in per.values()),
            macro_target_ccc=mean(r['ccc'] for r in per.values()), per_emotion=per)
    return result


if __name__ == '__main__':
    show(analyze())
