import torch
from .get_module import *
from .utils import *

def smooth_fn(As, Bs, p=2):
    s = torch.concat([normalize(B.weight) for B in Bs]).reshape(-1, Bs[0].weight.shape[-1]).abs().pow(p).mean(dim=0).pow(1/p)
    sa = s[:,None] if len(As[0].weight.shape) > 1 else s
    for A in As: A.weight.data.float().mul_(sa).to(A.weight.dtype)
    for B in Bs: B.weight.data.float().div_(s).to(B.weight.dtype)

def smooth_qkv(layer):
    norm = get_pre_norm(layer)
    qkv = [get_q(layer), get_k(layer), get_v(layer)]
    smooth_fn([norm], qkv)

def smooth_vo(layer):
    pass
    # 未実装

def smooth_mlp(layer):
    norm = get_post_norm(layer)
    up, gate, down = get_up(layer), get_gate(layer), get_down(layer)
    smooth_fn([up], [down])
    smooth_fn([norm], [up, gate])

def smooth_head(model):
    norm = get_head_norm(model)
    head = get_head(model)
    smooth_fn([norm], [head])

@torch.no_grad()
def apply_smooth(model):
    layers = get_layers(model)
    for l in layers:
        smooth_vo(l)
        smooth_qkv(l)
        smooth_mlp(l)
    smooth_head(model)