"""Paired changes in historical automated scores; not strict blind judgments."""
from statistics import mean
from .utils import ROOT, EMOTIONS, read_jsonl, show


def paired_contrasts(rows, cross=False):
    groups = {}
    for row in rows:
        if not row['alpha']:
            continue
        key = ((row['task_id'], row['prompt_language'], row['vector_language'], row['emotion'])
               if cross else (row['task_id'], row['kind'], row['emotion']))
        bucket = groups.setdefault(key, {})
        if row['alpha'] in bucket:
            raise ValueError('Duplicate condition: '+str(key))
        bucket[row['alpha']] = row
    contrasts = []
    for key, pair in groups.items():
        if set(pair) != {-0.5, 0.5}:
            raise ValueError('Missing endpoint: '+str(key))
        neg, pos = pair[-0.5], pair[0.5]
        emotion = pos['emotion']
        target = pos['judge'][emotion]-neg['judge'][emotion]
        other = mean(pos['judge'][e]-neg['judge'][e] for e in EMOTIONS if e != emotion)
        contrasts.append(dict(key=key, emotion=emotion, target_delta=target,
            selectivity=target-other, directional_hit=target > 0))
    return contrasts


def summarize(contrasts):
    return dict(n_pairs=len(contrasts), mean_delta=mean(r['target_delta'] for r in contrasts),
        mean_selectivity=mean(r['selectivity'] for r in contrasts),
        directional_hit=mean(r['directional_hit'] for r in contrasts))


def analyze():
    ordinary = paired_contrasts(read_jsonl(ROOT/'results/functionality/judged_responses.jsonl'))
    target = [r for r in ordinary if r['key'][1] == 'target']
    random = [r for r in ordinary if r['key'][1] == 'random']
    cross_rows = read_jsonl(ROOT/'results/crosslingual_functionality/judged_responses.jsonl')
    cross = paired_contrasts(cross_rows, cross=True)
    directions = {f'{vl}_vector_to_{pl}_prompt': summarize([r for r in cross
        if r['key'][1] == pl and r['key'][2] == vl]) for pl in ('en', 'zh') for vl in ('en', 'zh')}
    same = [r for r in cross if r['key'][1] == r['key'][2]]
    different = [r for r in cross if r['key'][1] != r['key'][2]]
    return dict(scoring_warning='Historical IDs exposed intervention conditions; not strictly blinded.',
        target=summarize(target), random=summarize(random),
        target_minus_random=mean(r['target_delta'] for r in target)-mean(r['target_delta'] for r in random),
        per_emotion={e: summarize([r for r in target if r['emotion'] == e]) for e in EMOTIONS},
        cross_language=dict(directions=directions, same=summarize(same), cross=summarize(different),
            cross_minus_same=mean(r['target_delta'] for r in different)-mean(r['target_delta'] for r in same)),
        chinese_quality=dict(n=sum(r['prompt_language'] == 'zh' for r in cross_rows),
            replacement_character_responses=sum(r['prompt_language'] == 'zh' and '\ufffd' in r['response'] for r in cross_rows)))


if __name__ == '__main__':
    show(analyze())
