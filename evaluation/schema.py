"""Normalize fresh inference and historical tables without silently mixing protocols."""
import math
from .utils import EMOTIONS


def validate_scores(rows, fields):
    for row in rows:
        for field in fields:
            if field not in row or not math.isfinite(float(row[field])):
                raise ValueError('Missing/nonfinite score: '+field+' in '+str(row.get('id')))


def readout_rows(rows, task):
    """Primary rows only. Call separately for each layer/pooling/vector-language."""
    rows = [dict(r) for r in rows]
    if not rows:
        raise ValueError('Empty readout table')
    for field in ('layer', 'pooling', 'dtype'):
        values = {str(r[field]) for r in rows if field in r}
        if len(values) > 1:
            raise ValueError('Mixed '+field+'; select a single protocol before scoring')
    if task == 'specificity':
        for r in rows:
            r.setdefault('condition', r.get('emotion'))
        validate_scores(rows, ['cos_'+e for e in EMOTIONS])
    elif task == 'sensitivity':
        for r in rows:
            for e in EMOTIONS:
                if 'mean_'+e not in r and 'cos_'+e in r:
                    r['mean_'+e] = r['cos_'+e]
        validate_scores(rows, ['mean_'+e for e in EMOTIONS])
    else:
        raise ValueError('Unsupported task: '+task)
    if any(r.get('version', 'original') != 'original' for r in rows):
        raise ValueError('Original rows required')
    keys = [(r['language'], r['id']) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate item-language readout')
    return rows


def invariance_rows(rows):
    """Pair long-form rows by ID, never by incidental file order."""
    if not rows:
        raise ValueError('Empty invariance table')
    for field in ('layer','pooling','dtype'):
        if len({str(r[field]) for r in rows if field in r})>1:
            raise ValueError('Mixed invariance '+field)
    if 'original_en_target' in rows[0]:
        validate_scores(rows, [v+'_'+l+'_target' for v in ('original','paraphrase') for l in ('en','zh')])
        return rows
    lookup = {}
    for r in rows:
        key = (r['id'], r['language'], r['version'])
        if key in lookup:
            raise ValueError('Duplicate invariance condition: '+str(key))
        lookup[key] = r
    result = []
    for item_id in sorted({r['id'] for r in rows}):
        group = []
        for lang in ('en','zh'):
            for version in ('original','paraphrase'):
                try:
                    group.append(lookup[item_id,lang,version])
                except KeyError as exc:
                    raise ValueError('Missing paired invariance condition: '+str(exc)) from exc
        first = group[0]
        if any((r['topic_id'],r['emotion'],str(r['level'])) !=
               (first['topic_id'],first['emotion'],str(first['level'])) for r in group):
            raise ValueError('Inconsistent paired metadata: '+item_id)
        out = {k:first[k] for k in ('id','topic_id','emotion','level')}
        for r in group:
            field = 'cos_'+r['emotion']
            validate_scores([r], [field])
            out[r['version']+'_'+r['language']+'_target'] = r[field]
        result.append(out)
    return result
