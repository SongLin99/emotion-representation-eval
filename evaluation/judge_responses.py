"""Optional new scoring run with anonymous shuffled payloads; costs API credits."""
import argparse
import json
import os
import random
from pathlib import Path
from urllib import request
from .utils import EMOTIONS, read_jsonl

MODEL = 'deepseek/deepseek-v4-flash-0731'
RUBRIC = '''Score each response using only the prompt and response. For happiness,
surprise, fear, disgust, anger, and sadness, assign integers 0..4 to the narrator's
language, appraisal, actions and tone: 0 absent, 1 faint, 2 clear, 3 strong, 4 dominant.
Multiple emotions may be nonzero. Score coherence, task_adherence and
language_adherence from 1..5. Penalize corrupted, mixed-language, or repetitive text.
Return JSON only: {"items":[{"id":"...","happiness":0,"surprise":0,"fear":0,
"disgust":0,"anger":0,"sadness":0,"coherence":5,"task_adherence":5,
"language_adherence":5}]}. Preserve every supplied ID exactly once.'''


def anonymous_payloads(rows, seed=20261001):
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    return [(index, dict(id=f'item_{slot:05d}', prompt=rows[index]['prompt'],
                        response=rows[index]['response'])) for slot, index in enumerate(order)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261001)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a versioned output filename; never overwrite old scores')
    key = os.environ.get('OPENROUTER_API_KEY')
    if not key:
        raise RuntimeError('Set OPENROUTER_API_KEY locally; do not commit credentials')
    rows = read_jsonl(args.input)
    blind = anonymous_payloads(rows, args.seed)
    judged = {}
    for start in range(0, len(blind), 20):
        batch = blind[start:start+20]
        payload = dict(model=MODEL, temperature=0, max_tokens=10000,
            reasoning={'enabled': False}, response_format={'type': 'json_object'},
            messages=[dict(role='user', content=RUBRIC+'\nDATA: '+json.dumps([p for _, p in batch], ensure_ascii=False))])
        req = request.Request('https://openrouter.ai/api/v1/chat/completions',
            data=json.dumps(payload).encode(), headers={'Authorization': 'Bearer '+key,
            'Content-Type': 'application/json'}, method='POST')
        with request.urlopen(req, timeout=300) as response:
            result = json.load(response)
        choice = result['choices'][0]
        if choice['finish_reason'] != 'stop':
            raise ValueError('Incomplete judge output')
        items = json.loads(choice['message']['content'])['items']
        mapped = {item['id']: item for item in items}
        if len(items) != len(batch) or len(mapped) != len(batch) or set(mapped) != {p['id'] for _, p in batch}:
            raise ValueError('Judge returned incorrect or duplicate IDs')
        for index, anonymous in batch:
            score = mapped[anonymous['id']]
            for emotion in EMOTIONS:
                if type(score[emotion]) is not int or not 0 <= score[emotion] <= 4:
                    raise ValueError('Invalid emotion score')
            for field in ('coherence', 'task_adherence', 'language_adherence'):
                if type(score[field]) is not int or not 1 <= score[field] <= 5:
                    raise ValueError('Invalid quality score')
            judged[index] = dict(rows[index], judge=score, judge_model=MODEL,
                                 judging_protocol='anonymous_shuffled_v1', judge_seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        for index in range(len(rows)):
            handle.write(json.dumps(judged[index], ensure_ascii=False)+'\n')


if __name__ == '__main__':
    main()
