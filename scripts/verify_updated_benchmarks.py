"""Recompute both corrected four-criterion summaries without GPU/API or archive writes."""
import contextlib
import io
import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.qwen_summary import compute
from evaluation.retest_summary import main as olmo_summary


def compare(actual, expected, path='summary'):
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and set(actual)==set(expected), path
        for k in expected:compare(actual[k], expected[k], path+'.'+k)
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual)==len(expected), path
        for i, (a, b) in enumerate(zip(actual, expected)):compare(a, b, f'{path}[{i}]')
    elif isinstance(expected, float):
        assert isinstance(actual, (int,float)) and math.isfinite(actual), path
        assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), path
    else:
        assert actual==expected, path


def main():
    with tempfile.TemporaryDirectory(prefix='emotion-benchmark-verify-') as d:
        target=Path(d)/'olmo_summary.json'
        with contextlib.redirect_stdout(io.StringIO()):olmo_summary(target)
        compare(json.loads(target.read_text()), json.loads((ROOT/'results/retest_20261004/summary.json').read_text()))
        print('PASS: complete corrected Olmo summary reproduced from item-level data', flush=True)
    compare(compute(), json.loads((ROOT/'results/qwen25_14b_20261004/summary.json').read_text()))
    print('PASS: complete Qwen summary reproduced, including coverage, bootstrap and all dose/quality conditions')


if __name__ == '__main__':main()
