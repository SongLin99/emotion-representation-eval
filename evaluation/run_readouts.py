"""Optional fresh inference on the frozen stories using supplied vectors."""
import argparse
import csv
from pathlib import Path
from .utils import ROOT, EMOTIONS, read_jsonl
from .model_utils import load_model, pool_hidden_states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['specificity', 'sensitivity', 'invariance'], required=True)
    parser.add_argument('--language', choices=['en', 'zh'], default='en')
    parser.add_argument('--layer', type=int, default=24, choices=range(33))
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output filename; frozen results must not be overwritten')
    if args.batch_size < 1:
        raise ValueError('batch-size must be positive')
    import torch
    from safetensors.numpy import load_file
    source = dict(specificity='specificity/stories.jsonl',
                  sensitivity='sensitivity/intensity_ladders.jsonl',
                  invariance='invariance/paraphrase_pairs.jsonl')[args.task]
    rows = read_jsonl(ROOT/'data'/source)
    vectors = torch.tensor(load_file(str(ROOT/'vectors/olmo3_7b_instruct'/args.language/
        'emotion_vectors.safetensors'))['unit_vectors'][args.layer], device=args.device)
    model, tokenizer = load_model(args.device, args.dtype)
    output = []
    fields = [('original', 'text_'+args.language)]
    if args.task == 'invariance':
        fields.append(('paraphrase', 'paraphrase_'+args.language))
    for version, text_field in fields:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start+args.batch_size]
            inputs = tokenizer([r[text_field] for r in batch], padding=True,
                               truncation=False, return_tensors='pt').to(args.device)
            if inputs['input_ids'].shape[1] > 2048:
                raise ValueError('Unexpected long input; no silent truncation')
            with torch.inference_mode():
                states = model.model(**inputs, output_hidden_states=True, use_cache=False).hidden_states
                if len(states) != 33:
                    raise ValueError('Checkpoint layer convention does not match the supplied vectors')
                pooled = pool_hidden_states(states[args.layer], inputs['attention_mask'])
                scores = (pooled@vectors.T)/pooled.norm(dim=-1, keepdim=True)
            for row, score in zip(batch, scores.cpu().tolist()):
                output.append(dict(id=row['id'], topic_id=row['topic_id'],
                    emotion=row.get('emotion', row.get('condition')), level=row.get('level', ''),
                    language=args.language, version=version, layer=args.layer,
                    dtype=args.dtype, **{'cos_'+e: x for e, x in zip(EMOTIONS, score)}))
            print(f'{version}: {min(start+args.batch_size, len(rows))}/{len(rows)}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)


if __name__ == '__main__':
    main()
