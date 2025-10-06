from .get_module import *
from .utils import *
import torch

# TODO
# permute_vo

@torch.no_grad()
def permute(A, perm):
    t = A.weight.shape
    A.weight.data = A.weight[..., perm]
    if hasattr(A, "act_scale"): A.act_scale = A.act_scale[..., perm]
    assert t == A.weight.shape

@torch.no_grad()
def permute_r(A, perm):
    t = A.weight.shape
    A.weight.data = A.weight[perm]
    assert t == A.weight.shape

def permute_embedding(model, perm):
    embed = get_embed(model)
    permute(embed, perm)

def permute_qkv(layer, perm):
    for e in [get_pre_norm(layer), get_q(layer), get_k(layer), get_v(layer)]:
        permute(e, perm)

@torch.no_grad()
def permute_vo(layer, perm_func=None):
    if perm_func is None: perm_func = get_perm
    v, o = get_v(layer), get_o(layer)
    dev = v.weight.device
    mv, mo = calc_metric(v, t=True), calc_metric(o)
    head_dim = get_head_dim(layer)
    ratio = mo.shape[0] // mv.shape[0]
    # metric = mv + mo.reshape(ratio,-1).mean(dim=0)
    metric = mo.reshape(ratio,-1).mean(dim=0)
    metric = metric.reshape(-1, head_dim)
    perm_v = torch.concat([perm_func(m) + head_dim * i for i, m in enumerate(metric)])
    perm_o = torch.concat([torch.concat([perm_func(m) + head_dim * (i * ratio + j) for j in range(ratio)]) for i, m in enumerate(metric)])
    permute_r(v, perm_v)
    permute(o, perm_o)
    # perm = get_perm_v2(metric.reshape(-1, head_dim).mean(dim=0))
    # permute_r(v, (perm[None].expand(mv.shape[0]//head_dim,-1) + head_dim * torch.arange(mv.shape[0]//head_dim).to(dev)[:,None]).reshape(-1))
    # permute(o, (perm[None].expand(mo.shape[0]//head_dim,-1) + head_dim * torch.arange(mo.shape[0]//head_dim).to(dev)[:,None]).reshape(-1))
    
def permute_o(layer, perm):
    o = get_o(layer)
    permute_r(o, perm)

def permute_mlp(layer, perm):
    for e in [get_post_norm(layer), get_up(layer), get_gate(layer)]:
        permute(e, perm)
    permute_r(get_down(layer), perm)

def permute_mlp_v2(layer, perm_func):
    up, gate, down = get_up(layer), get_gate(layer), get_down(layer)
    metric = calc_metric(down)
    perm = perm_func(metric)
    for e in [up, gate]: permute_r(e, perm)
    permute(down, perm)

def permute_head(model, perm):
    for e in [get_head_norm(model), get_head(model)]:
        permute(e, perm)

def get_perm(metric, k=32):
    idx = metric.argsort(dim=-1, descending=True)
    return idx

def get_perm_v2(metric, k=32):
    idx = metric.argsort(dim=-1)
    return idx

def get_perm_v3(metric, k=32):
    idx = metric.argsort(dim=-1, descending=True)
    n_group = idx.shape[0] // k
    tmp_idx = torch.arange(idx.shape[0])
    tmp_idx = tmp_idx % n_group * k + tmp_idx // n_group
    idx = idx[tmp_idx]
    return idx

def get_perm_v4(metric, k=96):
    idx = metric.argsort(dim=-1, descending=True)
    other, protect = idx[:-k], idx[-k:]
    idx = torch.concat([other.sort(dim=-1)[0], protect], dim=-1)
    return idx

@torch.no_grad()
def apply_permute(model, m=1):
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight)
    perm_func = [get_perm, get_perm_v2, get_perm_v3][m]
    layers = get_layers(model)
    for l in layers:
        permute_vo(l, perm_func)
        permute_mlp_v2(l, perm_func)

@torch.no_grad()
def apply_global_permute(model, perm):
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight)
    
    layers = get_layers(model)
    permute_embedding(model, perm)
    for l in layers:
        permute_qkv(l, perm)
        permute_o(l, perm)
        permute_mlp(l, perm)
    permute_head(model, perm)

@torch.no_grad()
def apply_global_permute_v2(model, m=0, n=0):
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight)
    
    perm_func = [get_perm, get_perm_v2, get_perm_v3, get_perm_v4][m]
    # emb = get_embed(model).weight
    # metric = emb.abs().max(dim=0)[0]
    # metric = get_head_norm(model).act_scale.abs()
    norm = get_head_norm(model)
    if n == 0: metric = norm.act_scale.abs().cpu() / norm.weight.abs()
    if n == 1: metric = norm.act_scale.abs().cpu()
    perm = perm_func(metric)
    apply_global_permute(model, perm)