"""Reader-facing labels and a shared model/language comparison table."""
import json
from .utils import ROOT


def comparison_table(language='en'):
    """Format frozen results without changing their calculation or stored schema."""
    summaries=[json.loads((ROOT/'results'/folder/'summary.json').read_text())
               for folder in ('retest_20261004','qwen25_14b_20261004')]
    zh=language=='zh'
    headers=(['指标','Olmo · 英文','Olmo · 中文','Qwen · 英文','Qwen · 中文'] if zh else
             ['Metric','Olmo · English','Olmo · Chinese','Qwen · English','Qwen · Chinese'])
    rows=[]
    metrics=[
        ('Emotion vs neutral (AUROC)','情绪与中性区分（AUROC）','specificity','target_neutral_macro',False),
        ('Emotion vs other emotions (AUROC)','不同情绪区分（AUROC）','specificity','cross_emotion_macro',False),
        ('Emotion classification accuracy','情绪分类准确率','specificity','top1',True),
        ('Intensity correlation (ρ)','强度相关（ρ）','sensitivity','mean_rho',False),
        ('Ladders increasing at every step','每一级都上升的阶梯比例','sensitivity','strict_monotonic',True),
        ('Paraphrase correlation (ρ)','改写相关（ρ）','invariance','macro_target_spearman',False)]
    for en,cn,key,field,percentage in metrics:
        cells=[cn if zh else en]
        for summary in summaries:
            primary=summary['primary']
            for lang in ('en','zh'):
                if key=='specificity':value=primary[key][lang][field]
                elif key=='sensitivity':value=primary[key][lang]['macro'][field]
                else:value=primary[key]['english_paraphrase' if lang=='en' else 'chinese_paraphrase'][field]
                cells.append(f'{100*value:.1f}%' if percentage else f'{value:.3f}')
        rows.append(cells)
    for limit,key in ((180,'functionality'),(320,'crosslingual_functionality')):
        effect=[f'Steer 效果（±0.5，上限{limit} token）' if zh else f'Steering effect (±0.5, {limit}-token limit)']
        count=[f'通过质量检查的回答配对（{limit} token）' if zh else f'Answer pairs passing quality checks ({limit}-token limit)']
        for summary in summaries:
            for lang in ('en','zh'):
                conditions=[x for x in summary[key]['conditions'] if x['kind']=='target'
                    and x['prompt_language']==x['vector_language']==lang and x['dose']==.5]
                if conditions:
                    c=conditions[0]
                    effect.append(f"{c['all_pairs']['mean_delta']:.3f}")
                    count.append(f"{c['valid_pairs']['n_pairs']}/{c['all_pairs']['n_pairs']}")
                else:effect.append('—');count.append('—')
        rows.extend([effect,count])
    return headers,rows
