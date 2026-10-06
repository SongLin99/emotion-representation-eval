"""Correct event-cluster uncertainty and independent response-quality checks."""
from collections import defaultdict
import numpy as np
from .utils import EMOTIONS

QUALITY_PROTOCOL=dict(replacement_character='invalid',token_limit='invalid',
    judge_minimum=3,minimum_requested_script_fraction=.5,
    scope='Predefined before inspecting new scores; show all pairs as well as valid pairs.')


def event_calibration(neutral,weights):
    """Weighted sample moments excluding all copies of each original event."""
    w=np.asarray(weights,dtype=float);neutral=np.asarray(neutral,dtype=float)
    n_cal=w.sum()-w
    if np.any(n_cal<=1):raise ValueError('Too few distinct calibration samples')
    mu=((w@neutral)[None,:]-w[:,None]*neutral)/n_cal[:,None]
    ss=(w@(neutral**2))[None,:]-w[:,None]*neutral**2-n_cal[:,None]*mu**2
    sd=np.sqrt(np.maximum(ss,0)/(n_cal[:,None]-1))
    return mu,sd


def specificity_bootstrap(rows,n=2000,seed=20261004):
    """Weighted event resampling; ALL copies of the focal event leave calibration."""
    topics=sorted({r['topic_id'] for r in rows});lookup={(r['topic_id'],r.get('condition',r.get('emotion'))):r for r in rows}
    conditions=(*EMOTIONS,'neutral')
    x=np.array([[[float(lookup[t,c]['cos_'+e]) for e in EMOTIONS] for c in conditions] for t in topics])
    neutral=x[:,6];rng=np.random.default_rng(seed);values=[]
    def measure(w):
        n_total=w.sum();mu,sd=event_calibration(neutral,w)
        if np.any(sd<=0):raise ValueError('Degenerate neutral calibration')
        z=(x[:,:6]-mu[:,None,:])/sd[:,None,:]
        maximum=z.max(2,keepdims=True);wins=np.abs(z-maximum)<=1e-12
        credit=wins[:,np.arange(6),np.arange(6)]/wins.sum(2)
        top1=(w[:,None]*credit).sum()/(n_total*6)
        aucs=[];cross=[]
        for e in range(6):
            for other in range(7):
                if e==other:continue
                a=x[:,e,e][:,None];b=x[:,other,e][None,:]
                comparison=(a>b)+.5*(a==b)
                value=float(w@comparison@w/(n_total**2))
                (aucs if other==6 else cross).append(value)
        return [float(np.mean(aucs)),float(np.mean(cross)),float(top1)]
    for _ in range(n):
        w=np.bincount(rng.integers(0,len(topics),len(topics)),minlength=len(topics));values.append(measure(w))
    point=measure(np.ones(len(topics)));values=np.asarray(values)
    return dict(n_bootstrap=n,unit='original_event',calibration='Exclude all sampled copies of current original event',
        metrics={k:dict(estimate=point[i],ci95=np.quantile(values[:,i],[.025,.975]).tolist())
                 for i,k in enumerate(('target_neutral_macro','cross_emotion_macro','top1'))})


def sensitivity_bootstrap(records,n=2000,seed=20261004):
    rng=np.random.default_rng(seed);fields=['rho','adjacent_accuracy','strict_monotonic','top1_credit']
    groups=[[r for r in records if r['emotion']==e] for e in EMOTIONS]
    if any(not x for x in groups):raise ValueError('All emotions required for stratified bootstrap')
    samples=[]
    for _ in range(n):
        sampled=[rs[i] for rs in groups for i in rng.integers(0,len(rs),len(rs))]
        samples.append([float(np.mean([r[f] for r in sampled])) for f in fields])
    values=np.asarray(samples)
    return dict(n_bootstrap=n,unit='event',strata='emotion',metrics={f:dict(
        estimate=float(np.mean([r[f] for r in records])),ci95=np.quantile(values[:,i],[.025,.975]).tolist()) for i,f in enumerate(fields)})


def response_quality(row):
    text=row['response'];judge=row['judge'];reasons=[]
    if '\ufffd' in text:reasons.append('replacement_character')
    if row.get('hit_token_limit'):reasons.append('token_limit')
    for key in ('coherence','task_adherence','language_adherence'):
        if judge.get(key,5)<3:reasons.append(key)
    han=sum('\u4e00'<=c<='\u9fff' for c in text)
    latin=sum(c.isascii() and c.isalpha() for c in text)
    lang=row.get('prompt_language','en')
    requested=han if lang=='zh' else latin
    if requested/max(1,han+latin)<.5:reasons.append('requested_script_fraction')
    return dict(valid=not reasons,reasons=reasons)


def cluster_interval(rows,field,n=2000,seed=20261004):
    if not rows:return None
    groups=defaultdict(list)
    for r in rows:groups[r['task_id']].append(float(r[field]))
    ids=sorted(groups);rng=np.random.default_rng(seed);vals=[]
    for _ in range(n):
        chosen=rng.choice(ids,len(ids),replace=True)
        vals.append(float(np.mean([v for t in chosen for v in groups[t]])))
    return np.quantile(vals,[.025,.975]).tolist()


def score_functionality(rows):
    """Both raw and quality-restricted estimates; never hide excluded endpoints."""
    groups=defaultdict(dict);baseline={}
    quality={r['id']:response_quality(r) for r in rows}
    for r in rows:
        pl=r.get('prompt_language','en');vl=r.get('vector_language','en');kind=r.get('kind','target')
        if not r['alpha']:
            baseline[r['task_id'],pl]=r;continue
        key=(r['task_id'],pl,vl,kind,r['emotion'],abs(r['alpha']))
        if r['alpha'] in groups[key]:raise ValueError('Duplicate signed condition')
        groups[key][r['alpha']]=r
    contrasts=[]
    for key,pair in sorted(groups.items()):
        task,pl,vl,kind,e,a=key
        if set(pair)!={-a,a}:raise ValueError('Missing signed endpoint: '+str(key))
        neg,pos=pair[-a],pair[a];b=baseline[task,pl]
        contrasts.append(dict(task_id=task,prompt_language=pl,vector_language=vl,kind=kind,emotion=e,dose=a,
            target_delta=pos['judge'][e]-neg['judge'][e],plus_vs_baseline=pos['judge'][e]-b['judge'][e],
            baseline_vs_minus=b['judge'][e]-neg['judge'][e],
            valid=quality[neg['id']]['valid'] and quality[pos['id']]['valid'],
            baseline_valid=quality[b['id']]['valid'],negative_id=neg['id'],positive_id=pos['id']))
    def summary(rs):
        return dict(n_pairs=len(rs),mean_delta=float(np.mean([r['target_delta'] for r in rs])) if rs else None,
            directional_hit=float(np.mean([r['target_delta']>0 for r in rs])) if rs else None,
            plus_vs_baseline=float(np.mean([r['plus_vs_baseline'] for r in rs])) if rs else None,
            baseline_vs_minus=float(np.mean([r['baseline_vs_minus'] for r in rs])) if rs else None,
            ci95=cluster_interval(rs,'target_delta'))
    conditions=[]
    for key in sorted({(r['prompt_language'],r['vector_language'],r['kind'],r['dose']) for r in contrasts}):
        pl,vl,kind,dose=key;rs=[r for r in contrasts if (r['prompt_language'],r['vector_language'],r['kind'],r['dose'])==key]
        valid=[r for r in rs if r['valid']]
        conditions.append(dict(prompt_language=pl,vector_language=vl,kind=kind,dose=dose,
            all_pairs=summary(rs),valid_pairs=summary(valid),valid_pair_coverage=len(valid)/len(rs)))
    paired_random=None
    if any(r['kind']=='random' for r in contrasts):
        control={(r['task_id'],r['emotion'],r['dose']):r for r in contrasts if r['kind']=='random'}
        differences=[]
        for r in contrasts:
            if r['kind']!='target':continue
            c=control[r['task_id'],r['emotion'],r['dose']]
            differences.append(dict(task_id=r['task_id'],target_delta=r['target_delta']-c['target_delta']))
        paired_random=dict(mean_difference=float(np.mean([r['target_delta'] for r in differences])),
            ci95=cluster_interval(differences,'target_delta'),n_pairs=len(differences))
    return dict(n_responses=len(rows),quality_protocol=QUALITY_PROTOCOL,conditions=conditions,contrasts=contrasts,
        target_minus_random=paired_random,
        quality=dict(n_invalid=sum(not x['valid'] for x in quality.values()),
            replacement_characters=sum('replacement_character' in x['reasons'] for x in quality.values()),
            items=quality),per_emotion={e:summary([r for r in contrasts if r['emotion']==e and r['kind']=='target']) for e in EMOTIONS})
