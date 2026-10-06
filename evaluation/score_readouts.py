"""End-to-end scoring CLI for frozen tables or fresh long-format inference."""
import argparse
import json
from pathlib import Path
from .utils import read_csv
from . import specificity, sensitivity, invariance


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--task', choices=('specificity','sensitivity','invariance'), required=True)
    p.add_argument('--input', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pooling',choices=('token50','continuation'))
    p.add_argument('--layer',type=int)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError('Never overwrite frozen results')
    rows=[r for file in a.input for r in read_csv(file)]
    if any('task' in r for r in rows):
        tasks=('sensitivity','invariance') if a.task=='invariance' else (a.task,)
        rows=[r for r in rows if r.get('task') in tasks]
    if a.pooling:rows=[r for r in rows if r.get('pooling')==a.pooling]
    if a.layer is not None:rows=[r for r in rows if str(r.get('layer'))==str(a.layer)]
    result={'task':a.task,'inputs':[str(x) for x in a.input],
            'metrics':{'specificity':specificity,'sensitivity':sensitivity,'invariance':invariance}[a.task].analyze(rows)}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
