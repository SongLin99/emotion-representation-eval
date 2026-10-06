"""One offline command turns the versioned re-test outputs into audited metrics."""
from pathlib import Path
import argparse,json
import numpy as np
from .utils import ROOT,EMOTIONS,read_csv,read_jsonl
from . import specificity,sensitivity,invariance
from .robustness import specificity_bootstrap,sensitivity_bootstrap,score_functionality

OUT=ROOT/'results/retest_20261004'
DATA=ROOT/'data/revisions/20261004'


def pair_accuracy(rows):
    lookup={(r['topic_id'],r.get('condition',r['emotion'])):r for r in rows}
    topics=sorted({r['topic_id'] for r in rows});per={}
    for e in EMOTIONS:
        deltas=[float(lookup[t,e]['cos_'+e])-float(lookup[t,'neutral']['cos_'+e]) for t in topics]
        per[e]=dict(n_pairs=len(deltas),accuracy=float(np.mean([1 if d>0 else .5 if d==0 else 0 for d in deltas])))
    return dict(macro=float(np.mean([x['accuracy'] for x in per.values()])),per_emotion=per)


def mapped_vector(rows,language):
    return [dict(r,**{'cos_'+e:r['cos_'+language+'_'+e] for e in EMOTIONS},vector_language=language) for r in rows]


def main(output=None):
    target=output or OUT/'summary.json'
    allrows=read_csv(OUT/'readouts_l24.csv')
    pick=lambda task,pool:[r for r in allrows if r['task']==task and r['pooling']==pool]
    sp=pick('specificity','token50');se=pick('sensitivity','token50');pa=pick('invariance','token50')
    primary=dict(specificity=specificity.analyze(sp),sensitivity=sensitivity.analyze(se),invariance=invariance.analyze(se+pa))
    corrected_original=dict(specificity=specificity.analyze(),sensitivity=sensitivity.analyze(),invariance=invariance.analyze())
    historical={name:json.loads((ROOT/f'results/{name}/summary.json').read_text()) for name in ('specificity','sensitivity','invariance')}
    intervals={};old_intervals={}
    for lang in ('en','zh'):
        group=[r for r in sp if r['language']==lang]
        primary['specificity'][lang]['matched_pair_accuracy']=pair_accuracy(group)
        intervals[lang]=dict(specificity=specificity_bootstrap(group),
            sensitivity=sensitivity_bootstrap(primary['sensitivity'][lang]['records']))
        old_intervals[lang]=specificity_bootstrap([r for r in read_csv(ROOT/'results/specificity/item_scores_l24.csv') if r['language']==lang])
    aligned_sp=pick('specificity','continuation');aligned_se=pick('sensitivity','continuation')
    robustness=dict(continuation_only=dict(specificity=specificity.analyze(aligned_sp),sensitivity=sensitivity.analyze(aligned_se)),
        fixed_vector_language={vl:dict(specificity=specificity.analyze(mapped_vector(sp,vl)),
            invariance=invariance.analyze(mapped_vector(se+pa,vl))) for vl in ('en','zh')})
    ordinary=score_functionality(read_jsonl(OUT/'functionality_judged.jsonl'))
    cross=score_functionality(read_jsonl(OUT/'crosslingual_judged.jsonl'))
    dose=score_functionality(read_jsonl(OUT/'small_dose_judged.jsonl'))
    quality_table=[]
    for file in ('crosslingual_judged.jsonl','small_dose_judged.jsonl'):
        rs=read_jsonl(OUT/file)
        for lang in ('en','zh'):
            for a in sorted({abs(r['alpha']) for r in rs}):
                xs=[r for r in rs if r['prompt_language']==lang and abs(r['alpha'])==a]
                if not xs:continue
                quality_table.append(dict(prompt_language=lang,dose=a,n=len(xs),replacement=sum('\ufffd' in r['response'] for r in xs),
                    truncated=sum(r.get('hit_token_limit',False) for r in xs),source=file))
    # Quantify any BF16 rerun differences on unchanged items, separately from material effects.
    original_material={r['id']:r for r in read_jsonl(ROOT/'data/sensitivity/intensity_ladders.jsonl')}
    new_material={r['id']:r for r in read_jsonl(DATA/'intensity_ladders.jsonl')}
    original_scores={(r['id'],r['language']):r for r in read_csv(ROOT/'results/sensitivity/item_scores_l24.csv')}
    residual=[]
    for r in se:
        if new_material[r['id']]['text_'+r['language']]==original_material[r['id']]['text_'+r['language']]:
            old=original_scores[r['id'],r['language']]
            residual.extend(abs(float(r['cos_'+e])-float(old['mean_'+e])) for e in EMOTIONS)
    result=dict(version='20261004_corrected',model='allenai/Olmo-3-7B-Instruct',primary_layer=24,
        scope='Six frozen emotion vectors; new L24 evaluation, not a new all-layer vector extraction.',
        primary=primary,corrected_statistics_on_original_data=corrected_original,historical=historical,
        intervals=intervals,corrected_original_specificity_intervals=old_intervals,robustness=robustness,
        functionality=ordinary,crosslingual_functionality=cross,small_dose_functionality=dose,
        generation_quality=quality_table,material_revision=json.loads((DATA/'final_materials_complete.json').read_text()),
        material_audit=json.loads((DATA/'semantic_audit.json').read_text()),
        unchanged_material_numerical_check=dict(n_scores=len(residual),max_absolute_error=float(np.max(residual)),mean_absolute_error=float(np.mean(residual))),
        scoring='All 2076 answers newly scored using anonymous shuffled IDs and independent objective quality checks. Rubric differs from historical scoring.',
        boundaries=['Materials and audits are AI-assisted, not human intensity ground truth.',
            'Quality-restricted contrasts are descriptive subsets; all raw contrasts and excluded cases remain available.',
            'Same-vector language transfer and aligned pooling are supplementary robustness analyses.',
            'A functional expression effect does not establish subjective emotion.'])
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({l:dict(auc=primary['specificity'][l]['target_neutral_macro'],top1=primary['specificity'][l]['top1'],
        rho=primary['sensitivity'][l]['macro']['mean_rho']) for l in ('en','zh')},indent=2))
    print('SUMMARY COMPLETE',target)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Use a new filename; never overwrite frozen results')
    main(args.output)
