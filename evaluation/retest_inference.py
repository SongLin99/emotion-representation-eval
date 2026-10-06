"""L24 versioned re-test: semantic repairs, aligned pooling, and smaller doses."""
from pathlib import Path
import argparse,csv,json,hashlib,time
from .utils import ROOT,EMOTIONS,read_jsonl
from .model_utils import load_model,pool_hidden_states,pool_continuation,assistant_intervention

DATA=ROOT/'data/revisions/20261004'
OUT=ROOT/'results/retest_20261004'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def readouts():
    import numpy as np
    import torch
    from safetensors.numpy import load_file
    files={'specificity':DATA/'specificity_stories.jsonl','sensitivity':DATA/'intensity_ladders.jsonl',
           'invariance':DATA/'paraphrase_pairs.jsonl'}
    signature={k:digest(v) for k,v in files.items()}
    completed=OUT/'readout_complete.json'
    if completed.exists():
        assert json.loads(completed.read_text())['inputs']==signature;print('READOUTS already complete',flush=True);return
    model,tok=load_model();capture={}
    units={lang:torch.tensor(load_file(str(ROOT/f'vectors/olmo3_7b_instruct/{lang}/emotion_vectors.safetensors'))['unit_vectors'][24],device='cuda') for lang in ('en','zh')}
    def hook(_m,_a,o):capture['h']=o[0] if isinstance(o,tuple) else o
    handle=model.model.layers[23].register_forward_hook(hook)
    out=[]
    try:
        for task,file in files.items():
            rows=read_jsonl(file)
            versions=['paraphrase'] if task=='invariance' else ['original']
            for lang in ('en','zh'):
                for version in versions:
                    field=('paraphrase_' if version=='paraphrase' else 'text_')+lang
                    for start in range(0,len(rows),12):
                        batch=rows[start:start+12]
                        encoded=tok([x[field] for x in batch],padding=True,return_offsets_mapping=True,return_tensors='pt')
                        spans=encoded.pop('offset_mapping');encoded=encoded.to('cuda')
                        with torch.inference_mode():
                            model.model(**encoded,use_cache=False,output_hidden_states=False)
                            h=capture.pop('h');mask=encoded['attention_mask']
                            pools={'token50':pool_hidden_states(h,mask)}
                            if version=='original':
                                assert all(x[field].startswith(x['shared_setup_'+lang]) for x in batch)
                                pools['continuation']=pool_continuation(h,mask,spans,[len(x['shared_setup_'+lang]) for x in batch])
                            for pooling,mean in pools.items():
                                normalized=mean/mean.norm(dim=-1,keepdim=True)
                                score={vl:(normalized@v.T).cpu().tolist() for vl,v in units.items()}
                                for i,x in enumerate(batch):
                                    record=dict(id=x['id'],topic_id=x['topic_id'],emotion=x.get('emotion',x.get('condition')),
                                        level=x.get('level',''),language=lang,version=version,task=task,layer=24,dtype='bfloat16',pooling=pooling,
                                        token_count=int(mask[i].sum()),vector_language=lang)
                                    for vl in ('en','zh'):
                                        record.update({'cos_'+vl+'_'+e:y for e,y in zip(EMOTIONS,score[vl][i])})
                                    record.update({'cos_'+e:y for e,y in zip(EMOTIONS,score[lang][i])})
                                    out.append(record)
                            del h,pools,encoded
                        if start%120==0:print('READOUT',task,lang,version,start,'/',len(rows),flush=True)
    finally:handle.remove()
    OUT.mkdir(parents=True,exist_ok=True)
    file=OUT/'readouts_l24.csv'
    with file.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    completed.write_text(json.dumps(dict(inputs=signature,n_rows=len(out),layer=24,poolings=['token50','continuation'],
        note='Frozen emotion vectors unchanged; both vector languages evaluated on every state.'),indent=2)+'\n')
    print('READOUTS COMPLETE',len(out),flush=True)


def doses():
    import numpy as np
    import torch
    from safetensors.numpy import load_file
    protocol=json.loads((ROOT/'data/functionality/crosslingual_protocol.json').read_text())
    doses=[.1,.25]
    path=OUT/'small_dose_responses.jsonl';OUT.mkdir(parents=True,exist_ok=True)
    existing=read_jsonl(path) if path.exists() else [];done={x['id'] for x in existing}
    complete=OUT/'doses_complete.json'
    if complete.exists():assert len(existing)==1176;print('DOSES already complete',flush=True);return
    norm=protocol.get('fixed_norm',protocol.get('fixed_neutral_token_norm'));layer=24
    units={lang:load_file(str(ROOT/f'vectors/olmo3_7b_instruct/{lang}/emotion_vectors.safetensors'))['unit_vectors'][24] for lang in ('en','zh')}
    model,tok=load_model();eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
    with path.open('a',encoding='utf-8') as f:
        for task in protocol['tasks']:
            for lang in ('en','zh'):
                prompt=task['prompt_'+lang]
                for dose in doses:
                    jobs=[]
                    if dose==doses[0]:jobs.append(dict(id=f'{lang}_{task["task_id"]}_baseline',task_id=task['task_id'],prompt_language=lang,
                        vector_language='none',emotion='none',alpha=0.,prompt=prompt,kind='baseline'))
                    for vl in ('en','zh'):
                        for e in EMOTIONS:
                            for sign in (-1,1):
                                a=sign*dose
                                jobs.append(dict(id=f'{lang}_{task["task_id"]}_{vl}_{e}_{a:+.2f}',task_id=task['task_id'],prompt_language=lang,
                                    vector_language=vl,emotion=e,alpha=a,prompt=prompt,kind='target'))
                    jobs=[x for x in jobs if x['id'] not in done]
                    if not jobs:continue
                    # Microbatches limit KV memory and are recorded; no user tokens are edited.
                    for start in range(0,len(jobs),16):
                        batch=jobs[start:start+16]
                        texts=[f'<|im_start|>user\n{x["prompt"]}<|im_end|>\n<|im_start|>assistant\n' for x in batch]
                        inp=tok(texts,padding=True,add_special_tokens=False,return_tensors='pt').to('cuda')
                        delta=torch.tensor(np.stack([x['alpha']*norm*units[x['vector_language']][EMOTIONS.index(x['emotion'])]
                            if x['alpha'] else np.zeros(4096,dtype='float32') for x in batch]),device='cuda')
                        with assistant_intervention(model,delta,layer),torch.inference_mode():
                            gen=model.generate(**inp,do_sample=False,use_cache=True,max_new_tokens=320,pad_token_id=tok.pad_token_id)
                        for x,seq in zip(batch,gen):
                            ids=seq[inp['input_ids'].shape[1]:].tolist();end=next((i for i,k in enumerate(ids) if k in eos),len(ids));ids=ids[:end]
                            row=dict(x,response=tok.decode(ids,skip_special_tokens=True).strip(),generated_token_ids=ids,generated_tokens=end,
                                hit_token_limit=end==320,layer=24,fixed_norm=norm,requested_delta_norm=abs(x['alpha'])*norm,
                                inference_batch_size=len(batch),protocol='small_dose_assistant_only_v2')
                            f.write(json.dumps(row,ensure_ascii=False)+'\n');done.add(row['id'])
                        f.flush();print('DOSE',task['task_id'],lang,dose,start,'responses',len(done),flush=True)
    assert len(done)==1176
    complete.write_text(json.dumps(dict(n_responses=1176,doses=doses,n_prompts=12,prompt_languages=['en','zh'],
        vector_languages=['en','zh'],layer=24,fixed_norm=norm,max_new_tokens=320,greedy=True),indent=2)+'\n')
    print('DOSES COMPLETE',len(done),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('readouts','doses'));a=p.parse_args()
    readouts() if a.phase=='readouts' else doses()
