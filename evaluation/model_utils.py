"""Model evaluation only; no vector construction or PCA implementation."""
from contextlib import contextmanager

MODEL_ID = 'allenai/Olmo-3-7B-Instruct'
REVISION = '096bb5469fe34348bc88d851a69edb3bf6f40df4'


def load_model(device='cuda', dtype='bfloat16'):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION,
        torch_dtype=getattr(torch, dtype), attn_implementation='sdpa').to(device).eval()
    model.requires_grad_(False)
    return model, tokenizer


def pool_hidden_states(hidden, mask, offset=49):
    """Include token 50 onward, counted separately among valid tokens per row."""
    import torch
    valid = mask.to(device=hidden.device, dtype=torch.bool)
    keep = valid & (valid.long().cumsum(1) > offset)
    count = keep.sum(1)
    if not bool((count > 0).all()):
        raise ValueError('Story has fewer than 50 valid tokens')
    return (hidden.float()*keep.unsqueeze(-1)).sum(1)/count.unsqueeze(-1)


def pool_continuation(hidden, mask, offsets, starts):
    """Pool tokens overlapping the continuation, aligned by character boundaries."""
    import torch
    spans=torch.as_tensor(offsets,device=hidden.device)
    begin=torch.as_tensor(starts,device=hidden.device)[:,None]
    keep=mask.bool() & (spans[:,:,1]>begin) & (spans[:,:,1]>spans[:,:,0])
    counts=keep.sum(1)
    if not bool((counts>0).all()):raise ValueError('No continuation tokens')
    return (hidden.float()*keep[:,:,None]).sum(1)/counts[:,None]


@contextmanager
def assistant_intervention(model, delta, layer=24):
    """Edit only the last prefill position and each subsequent cached decode state."""
    if not 1<=layer<=len(model.model.layers):
        raise ValueError('Intervention layer must index an actual transformer block')
    def hook(_module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        edited = hidden.clone()
        edited[:, -1] = hidden[:, -1] + delta.to(hidden.device, hidden.dtype)
        return (edited, *output[1:]) if isinstance(output, tuple) else edited
    handle = model.model.layers[layer-1].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
