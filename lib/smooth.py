import torch
from .get_module import *
from .utils import *

@torch.no_grad()
def _smooth_fn(As, Bs, p=2, a=0., b=0.5):
    sa = torch.concat([normalize(A.weight)[..., None] for A in As]).reshape(As[0].weight.shape[0], -1).abs().pow(p).mean(dim=1).pow(1/p)
    sb = torch.concat([normalize(B.weight) for B in Bs]).reshape(-1, Bs[0].weight.shape[-1]).abs().pow(p).mean(dim=0).pow(1/p)
    s = sa.pow(-a) * sb.pow(b)
    s_ = s[:,None] if len(As[0].weight.shape) > 1 else s
    for A in As: A.weight.data = A.weight.float().mul_(s_).to(A.weight.dtype)
    for B in Bs: B.weight.data = B.weight.float().div_(s).to(B.weight.dtype)

def smooth_fn(As, Bs, n_iterations=100, lr=0.01, a=None, b=None):
    s = torch.nn.Parameter(torch.ones(Bs[0].weight.shape[-1], device=Bs[0].weight.device))
    optimizer = torch.optim.Adam([s], lr=lr)
    for i in range(n_iterations):
        optimizer.zero_grad()
        loss = 0
        if len(As[0].weight.shape) > 1:
            # for A in As: loss += quantization_loss(A.weight * s[:,None])
            for B in Bs: loss += quantization_loss(B.weight / s)
        else:
            for B in Bs: loss += quantization_loss(B.weight / s, norm=As[0])
        loss.backward()
        optimizer.step()
        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1}/{n_iterations}, Loss: {loss.item():.6f}")
    s_ = s[:,None] if len(As[0].weight.shape) > 1 else s
    for A in As: A.weight.data = A.weight.float().mul_(s_).to(A.weight.dtype)
    for B in Bs: B.weight.data = B.weight.float().div_(s).to(B.weight.dtype)

def smooth_qkv(layer, a, b):
    norm = get_pre_norm(layer)
    qkv = [get_q(layer), get_k(layer), get_v(layer)]
    smooth_fn([norm], qkv, a=a, b=b)

@torch.no_grad()
def smooth_vo(layer, a=0.5, b=0.5):
    head_dim = get_head_dim(layer)
    v, o = get_v(layer), get_o(layer)
    w_o, w_v = o.weight.data, v.weight.data
    tmp = w_o.shape
    ratio = w_o.shape[1] // w_v.shape[0]
    w_o = w_o.reshape(w_o.shape[0],-1,ratio,head_dim).transpose(1,2).reshape(ratio*w_o.shape[0],-1)
    
    p = 2
    s_v = normalize(w_v).abs().pow(p).mean(dim=1).pow(1/p)
    s_o = normalize(w_o).abs().pow(p).mean(dim=0).pow(1/p)
    s = s_v.pow(a) / s_o.pow(b)

    v.weight.data = v.weight.div(s[:,None]).to(w_v.dtype)
    o.weight.data = w_o.mul(s).reshape(-1,ratio,w_o.shape[1]//head_dim,head_dim).transpose(1,2).reshape(tmp).to(w_o.dtype)

def smooth_mlp(layer, a, b):
    norm = get_post_norm(layer)
    up, gate, down = get_up(layer), get_gate(layer), get_down(layer)
    smooth_fn([up], [down], a=a, b=b)
    smooth_fn([norm], [up, gate], a=a, b=b)

def smooth_head(model, a, b):
    norm = get_head_norm(model)
    head = get_head(model)
    smooth_fn([norm], [head], a=a, b=b)

def apply_smooth(model, a=0., b=0.5, device=None):
    device = get_device()
    model.cpu()
    layers = get_layers(model)
    for l in layers:
        l.to(device)
        # smooth_vo(l)
        smooth_qkv(l, a, b)
        smooth_mlp(l, a, b)
        l.cpu()
    smooth_head(model, a, b)