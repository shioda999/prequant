import torch
from .get_module import *

@torch.no_grad()
def normalize(A):
    return A.float().div(A.float().pow(2).mean().sqrt())

def random_rotation_matrix(dim: int, device):
    A = torch.randn(dim, dim)
    Q, R = torch.linalg.qr(A)
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q.to(device)

def divide(model):
    layers = get_layers(model)
    head_dim = get_head_dim(model)
    num_heads = get_num_heads(model)
    num_kv_heads = get_num_kv_heads(model)
    for l in layers:
        attn, mlp = l.self_attn, l.mlp
        if hasattr(attn, "qkv_proj"):
            model.qkv_merge = True
            
            w_qkv = attn.qkv_proj.weight
            o_dim, i_dim = w_qkv.shape
            
            attn.q_proj = torch.nn.Linear(i_dim, head_dim * num_heads, False, w_qkv.device, w_qkv.dtype)
            attn.k_proj = torch.nn.Linear(i_dim, head_dim * num_kv_heads, False, w_qkv.device, w_qkv.dtype)
            attn.v_proj = torch.nn.Linear(i_dim, head_dim * num_kv_heads, False, w_qkv.device, w_qkv.dtype)

            attn.q_proj.weight.data = w_qkv[:head_dim * num_heads]
            attn.k_proj.weight.data = w_qkv[head_dim * num_heads:head_dim * (num_kv_heads + num_heads)]
            attn.v_proj.weight.data = w_qkv[head_dim * (num_kv_heads + num_heads):]

            del attn.qkv_proj

            attn.qkv_proj = ConcatModule(attn.q_proj, attn.k_proj, attn.v_proj)

        if hasattr(mlp, "gate_up_proj"):
            model.gate_up_merge = True
            w_gate_up = mlp.gate_up_proj.weight
            o_dim, i_dim = w_gate_up.shape
            mlp.gate_proj = torch.nn.Linear(i_dim, o_dim // 2, False, w_gate_up.device, w_gate_up.dtype)
            mlp.up_proj = torch.nn.Linear(i_dim, o_dim // 2, False, w_gate_up.device, w_gate_up.dtype)
            
            mlp.gate_proj.weight.data = w_gate_up[:o_dim // 2]
            mlp.up_proj.weight.data = w_gate_up[o_dim // 2:]
            
            del mlp.gate_up_proj
            mlp.gate_up_proj = ConcatModule(mlp.gate_proj, mlp.up_proj)

def undivide(model):
    layers = get_layers(model)
    head_dim = get_head_dim(model)
    num_heads = get_num_heads(model)
    num_kv_heads = get_num_kv_heads(model)
    for l in layers:
        attn, mlp = l.self_attn, l.mlp
        if hasattr(attn, "qkv_merge") and model.qkv_merge:
            del attn.qkv_proj
            w_q = attn.q_proj.weight
            o_dim, i_dim = w_q.shape
            attn.qkv_proj = torch.nn.Linear(i_dim, head_dim * (num_heads + num_kv_heads * 2),
                                         False, w_q.device, w_q.dtype)
            attn.qkv_proj.weight.data = torch.concat([
                attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight
            ])

            del attn.q_proj, attn.k_proj, attn.v_proj

        if hasattr(mlp, "gate_up_merge") and model.gate_up_merge:
            del mlp.gate_up_proj
            w_gate = mlp.gate_proj.weight
            o_dim, i_dim = w_gate.shape
            mlp.gate_up_proj = torch.nn.Linear(i_dim, o_dim * 2, False, w_gate.device, w_gate.dtype)
            mlp.gate_up_proj.weight.data = torch.concat([
                mlp.gate_proj.weight,
                mlp.up_proj.weight,
            ])

            del mlp.gate_proj, mlp.up_proj

class ConcatModule:
    def __init__(self, *modules):
        self.modules = modules
    def __call__(self, x):
        y = torch.concat([m(x) for m in self.modules], dim=-1)
        return y

def quantize(w, nbits=4, group_sz=128):
    shape, dtype = w.shape, w.dtype
    w = w.reshape(-1, group_sz).float()
    Qp, Qn = 2 ** (nbits - 1) - 1, -2 ** (nbits - 1)
    s = torch.maximum(w.max(dim=1, keepdim=True)[0] / Qp, w.min(dim=1, keepdim=True)[0] / Qn)
    w_q = w.div(s).round_().clamp_(Qn, Qp).mul_(s).reshape(shape).to(dtype)
    return w_q, s

def q_err(m, nbits=4, group_sz=32, norm=None, t=False):
    w = m.weight
    w_q, s = quantize(w, nbits, group_sz)
    delta = w_q - w
    if norm is not None:
        delta.mul_(norm.weight)
    if t is False:
        return delta.reshape(w.shape[0],-1, group_sz).float().pow(2).mean(dim=-1).mean(dim=0)
    else:
        return delta.reshape(group_sz, -1).float().pow(2).mean(dim=-1)

def calc_quantize_error(model):
    result = {"!SUM": 0}

    def register(m, labels, nbits=4, norm=None, t=False):
        err = q_err(m, nbits, norm=norm, t=t)
        result["!SUM"] += err.mean()
        for e in labels:
            if e not in result: result[e] = err
            elif result[e].shape != err.shape: result[e] = result[e].sum() + err.mean()
            else: result[e] += err

    register(get_embed(model), ["embed"])
    layers = get_layers(model)
    for i, l in enumerate(layers):
        pre_norm, post_norm = get_pre_norm(l), get_post_norm(l)
        register(get_q(l), ["q", f"{i:02}"], norm=pre_norm)
        register(get_k(l), ["k", f"{i:02}"], norm=pre_norm)
        register(get_v(l), ["v", f"{i:02}"], 6, norm=pre_norm)
        register(get_o(l), ["o", f"{i:02}"], t=True)
        register(get_gate(l), ["gate", f"{i:02}"], norm=post_norm)
        register(get_up(l), ["up", f"{i:02}"], norm=post_norm)
        register(get_down(l), ["down", f"{i:02}"], 6, t=True)
    register(get_head(model), ["head"], norm=get_head_norm(model))

    return result
    

@torch.no_grad()
def fuse_norm(norm, fcs):
    for fc in fcs:
        setattr(fc, "prev_dtype", fc.weight.dtype)
        fc.weight.data = norm.weight.float() * fc.weight.float()
    setattr(norm, "prev_weight", norm.weight.data.clone())
    norm.weight.data = torch.ones_like(norm.weight, dtype=norm.weight.dtype)

@torch.no_grad()
def _defuse_norm(norm, fcs, p=2):
    # s = norm.prev_weight.reshape(sz, -1).abs().mean(dim=0)[None].expand((sz, -1)).reshape(-1).sqrt()
    s = torch.concat([normalize(fc.weight.data) for fc in fcs]).reshape(-1, fcs[0].weight.shape[-1]).abs().pow(p).mean(dim=0).pow(1/p).pow(0.5)
    for fc in fcs:
        fc.weight.data = fc.weight.data.float().div(s).to(fc.prev_dtype)
        del fc.prev_dtype
    norm.weight.data = norm.weight.data.float().mul(s).to(norm.weight.dtype)
    del norm.prev_weight

def safe_divide(numerator, denominator, eps=1e-6):
    return numerator / (denominator + (denominator.abs() < eps) * eps)

@torch.no_grad()
def defuse_norm(norm, fcs):
    for fc in fcs:
        fc.weight.data = safe_divide(fc.weight.float(), norm.prev_weight.float()).to(fc.prev_dtype)
        del fc.prev_dtype
    norm.weight.data = norm.prev_weight.to(norm.weight.dtype)
    del norm.prev_weight
    

@torch.no_grad()
def mean_norm(norm, H):
    t = H.T.abs() @ norm.prev_weight.abs().float() * norm.prev_weight.float().sign()
    norm.prev_weight = t

@torch.no_grad()
def apply_quantize(model):
    for l in get_layers(model):
        for m in [get_q(l), get_k(l), get_o(l),
                   get_gate(l), get_up(l)]:
            m.weight.data, _ = quantize(m.weight)
        for m in [get_v(l), get_down(l)]:
            m.weight.data, _ = quantize(m.weight, 6)
    
    for m in [get_embed(model), get_head(model)]:
        m.weight.data, _ = quantize(m.weight)

@torch.no_grad()
def calc_metric(m, t=False):
    w = m.weight if not t else m.weight.T
    t = w.abs().pow(2).mean(dim=0).sqrt()
    return t / t.mean()
