"""Optional assistant-only steering; writes new responses, never old results."""
import argparse
import json
from pathlib import Path
from .utils import ROOT, EMOTIONS
from .model_utils import load_model, assistant_intervention


def make_jobs(task, cross=False):
    prompts = [(lang, task['prompt_'+lang]) for lang in ('en', 'zh')] if cross else [('en', task['prompt'])]
    for language, prompt in prompts:
        yield dict(id=f'{language}_{task["task_id"]}_baseline', task_id=task['task_id'],
            prompt=prompt, prompt_language=language, vector_language='none',
            kind='baseline', emotion='none', alpha=0.0)
        for vector_language in (('en', 'zh') if cross else ('en',)):
            for kind in (('target',) if cross else ('target', 'random')):
                for emotion in EMOTIONS:
                    for alpha in (-0.5, 0.5):
                        yield dict(id=f'{language}_{task["task_id"]}_{vector_language}_{kind}_{emotion}_{alpha:+.1f}',
                            task_id=task['task_id'], prompt=prompt, prompt_language=language,
                            vector_language=vector_language, kind=kind, emotion=emotion, alpha=alpha)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cross-language', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output path')
    import numpy as np
    import torch
    from safetensors.numpy import load_file
    name = 'crosslingual_protocol.json' if args.cross_language else 'functionality_protocol.json'
    protocol = json.loads((ROOT/'data/functionality'/name).read_text())
    layer = protocol['layer']
    norm = protocol.get('fixed_norm', protocol.get('fixed_neutral_token_norm'))
    units = {language: load_file(str(ROOT/'vectors/olmo3_7b_instruct'/language/
        'emotion_vectors.safetensors'))['unit_vectors'][layer] for language in ('en', 'zh')}
    random = np.random.default_rng(20260930).normal(size=units['en'].shape).astype('float32')
    _, singular, vt = np.linalg.svd(units['en'], full_matrices=False)
    basis = vt[singular > 1e-6]
    random -= random@basis.T@basis
    random /= np.linalg.norm(random, axis=1, keepdims=True)
    model, tokenizer = load_model(args.device)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        for task in protocol['tasks']:
            for language in (('en', 'zh') if args.cross_language else ('en',)):
                batch = [j for j in make_jobs(task, args.cross_language) if j['prompt_language'] == language]
                # Frozen native template: no system message, no extra BOS.
                texts = [f'<|im_start|>user\n{j["prompt"]}<|im_end|>\n<|im_start|>assistant\n' for j in batch]
                inputs = tokenizer(texts, padding=True, add_special_tokens=False, return_tensors='pt').to(args.device)
                deltas = []
                for job in batch:
                    vector = (random if job['kind'] == 'random' else units.get(job['vector_language'], units['en']))
                    direction = vector[EMOTIONS.index(job['emotion'])] if job['alpha'] else np.zeros(4096, dtype='float32')
                    deltas.append(job['alpha']*norm*direction)
                delta = torch.tensor(np.stack(deltas), device=args.device)
                with assistant_intervention(model, delta, layer), torch.inference_mode():
                    generated = model.generate(**inputs, do_sample=False, use_cache=True,
                        max_new_tokens=protocol['max_new_tokens'], pad_token_id=tokenizer.pad_token_id)
                for job, sequence in zip(batch, generated):
                    tokens = sequence[inputs['input_ids'].shape[1]:].tolist()
                    end = next((i for i, token in enumerate(tokens) if token in eos), len(tokens))
                    response = dict(job, response=tokenizer.decode(tokens[:end], skip_special_tokens=True),
                        generated_tokens=end, hit_token_limit=end == protocol['max_new_tokens'],
                        generated_token_ids=tokens[:end], layer=layer, fixed_norm=norm)
                    handle.write(json.dumps(response, ensure_ascii=False)+'\n')
                handle.flush()
                print(task['task_id'], language, 'complete', flush=True)


if __name__ == '__main__':
    main()
