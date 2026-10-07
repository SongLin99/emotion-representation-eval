"""Apply the user-requested semantic quality policy to existing frozen answers."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from .utils import ROOT, read_jsonl
from .robustness import QUALITY_PROTOCOL, score_functionality

OUT=ROOT/'results/quality_revision_20261007'
SOURCES={'olmo3_7b':'retest_20261004','qwen25_14b':'qwen25_14b_20261004'}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def compute(model):
    source=ROOT/'results'/SOURCES[model]
    result=copy.deepcopy(json.loads((source/'summary.json').read_text()))
    inputs={'summary.json':sha(source/'summary.json')}
    changes=[]
    for dataset,key in [('functionality','functionality'),('crosslingual','crosslingual_functionality'),('small_dose','small_dose_functionality')]:
        path=source/(dataset+'_judged.jsonl')
        inputs[path.name]=sha(path)
        old=result[key]
        fresh=score_functionality(read_jsonl(path),QUALITY_PROTOCOL)
        before={r['id']:r for r in read_jsonl(path)}
        for identifier,q in fresh['quality']['items'].items():
            previous=old['quality']['items'][identifier]
            if previous!=q:
                changes.append(dict(dataset=dataset,id=identifier,old_reasons=previous['reasons'],
                    new_reasons=q['reasons'],old_valid=previous['valid'],new_valid=q['valid'],
                    hit_token_limit=before[identifier].get('hit_token_limit',False)))
        result[key]=fresh
    if model=='qwen25_14b':
        reference=compute('olmo3_7b')
        for k in ('functionality','crosslingual_functionality','small_dose_functionality'):
            result['olmo_reference'][k]=reference[k]['conditions']
    result['quality_revision']=dict(version='semantic_quality_20261007',source_directory='results/'+SOURCES[model],
        source_sha256=inputs,policy=QUALITY_PROTOCOL,changes=changes,
        interpretation='No new inference or API judging. Existing semantic quality ratings retained; cap alone allowed; objective repetition guard added.',
        original_archive_commit='efb527cdfa969584f58b5ab79639b3eeaf774d49')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    args=parser.parse_args()
    for model in SOURCES:
        target=args.output_dir/model/'summary.json'
        if target.exists():raise FileExistsError('Use a new output directory; never overwrite archived revisions')
    for model in SOURCES:
        target=args.output_dir/model/'summary.json'
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('x',encoding='utf-8') as handle:
            handle.write(json.dumps(compute(model),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        print('QUALITY REVISION COMPLETE',model,target,flush=True)


if __name__=='__main__':main()
