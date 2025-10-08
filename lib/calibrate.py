import torch
from .get_module import *
from .utils import *
from functools import partial
from datasets import load_from_disk

def get_default_dataset(tokenizer):
    testdata = load_from_disk("./data/wikitext_test")
    return tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

@torch.no_grad()
def stat_act(model, tokenizer, dataset=None, num_samples=10, seq_len=None):
    if dataset is None: dataset = get_default_dataset(tokenizer)
    if seq_len is None: seq_len = model.seqlen
    prev_device = model.device
    device = get_device()
    model.to(device)
    layers = get_layers(model)
    act_scales = {}
    cnt = {}
    
    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_l2 = tensor.abs().double().pow(2).mean(dim=0).sqrt().float()
        if name in act_scales:
            act_scales[name] = (act_scales[name].double().pow(2)*cnt[name]/(cnt[name]+1) + comming_l2.double().pow(2)/(cnt[name]+1)).sqrt().float()
            cnt[name] += 1
        else:
            act_scales[name] = comming_l2
            cnt[name] = 1

    
    def stat_input_hook(m, x, y, name):
        stat_tensor(name, x[0] if isinstance(x, tuple) else x)
    
    hooks = []
    target_class = (torch.nn.Linear, get_head_norm(model).__class__)
    for name, m in model.named_modules():
        if isinstance(m, target_class):
            hooks.append(
                m.register_forward_hook(partial(stat_input_hook, name=name))
            )
            # print(name)

    if hasattr(dataset, 'input_ids'):
        inputs = [dataset.input_ids[:,i:i+seq_len] for i in range(0,dataset.input_ids.shape[1],seq_len)]
    else:
        inputs = []
        for data in dataset:
            if len(data["text"]) > 0:
            # if len(data["text"]) > 50:
                input_ids = tokenizer(
                    data["text"], return_tensors="pt", max_length=seq_len, truncation=True
                ).input_ids
                inputs.append(input_ids)

    inputs = inputs[:num_samples]
    bar = tqdm(total=len(inputs))
    for input_ids in inputs:
        model(input_ids.to(device))
        bar.update(1)

    for h in hooks:
        h.remove()

    for name, m in model.named_modules():
        if isinstance(m, target_class):
            t = act_scales[name]
            t = t / t.abs().mean()
            t = torch.clamp(t, 0.25, 4.)
            m.act_scale = t

    model.to(prev_device)
    return act_scales