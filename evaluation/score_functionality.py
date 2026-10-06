"""Score supplied anonymous judgments, including signed dose and quality strata."""
import argparse,json
from pathlib import Path
from .utils import read_jsonl
from .robustness import score_functionality

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Never overwrite frozen summaries')
    result=score_functionality(read_jsonl(a.input));a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
