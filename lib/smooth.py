import torch
from .get_module import *
from .utils import *
import math
import random

@torch.no_grad()
def _smooth_fn(As, Bs, p=2, a=0., b=0.5):
    sa = torch.concat([normalize(A.weight)[..., None] for A in As]).reshape(As[0].weight.shape[0], -1).abs().pow(p).mean(dim=1).pow(1/p)
    sb = torch.concat([normalize(B.weight) for B in Bs]).reshape(-1, Bs[0].weight.shape[-1]).abs().pow(p).mean(dim=0).pow(1/p)
    s = sa.pow(-a) * sb.pow(b)
    s_ = s[:,None] if len(As[0].weight.shape) > 1 else s
    for A in As: A.weight.data = A.weight.float().mul_(s_).to(A.weight.dtype)
    for B in Bs: B.weight.data = B.weight.float().div_(s).to(B.weight.dtype)

def quantization_loss_for_smooth(w, group_sz=32, nbits=4, scale=None):
    shape = w.shape
    w = (w / scale).reshape(-1, group_sz)
    Qp, Qn = 2 ** (nbits - 1) - 1, -2 ** (nbits - 1)
    s = torch.maximum(w.max(dim=1, keepdim=True)[0] / Qp, w.min(dim=1, keepdim=True)[0] / Qn)#.detach()
    w_q = torch.round(w.div(s)).clamp(Qn, Qp).mul(s)
    delta = w - w_q
    delta = delta.reshape(shape).mul(scale)
    return delta.pow(2)

def bincount_ste(t):
    result = torch.zeros((int(t.max() + 3),))
    ones = torch.ones_like(t) + (t - t.detach())
    result = torch.scatter_reduce(result, 0, t.to(torch.long), ones, "sum")
    result = torch.scatter_reduce(result, 0, t.to(torch.long) + 1, ones * 5, "sum")
    result = torch.scatter_reduce(result, 0, t.to(torch.long) + 2, ones, "sum")
    return result

def _quantization_loss_for_smooth(w, group_sz=32, nbits=4, scale=None):
    shape = w.shape
    w = (w / scale).reshape(-1, group_sz)
    Qp, Qn = 2 ** (nbits - 1) - 1, -2 ** (nbits - 1)
    s = torch.maximum(w.max(dim=1, keepdim=True)[0] / Qp, w.min(dim=1, keepdim=True)[0] / Qn)
    w_q = round_ste(w.div(s)).clamp(Qn, Qp)
    delta = w - w_q.mul(s)
    delta = delta.reshape(shape).mul(scale)

    # bin = torch.bincount((w_q - Qn).reshape(-1))
    bin = bincount_ste((w_q - Qn).reshape(-1))
    p = bin / w.numel()
    entropy = (-p * p.log()).sum()

    grad = -entropy
    # return -entropy
    return delta.pow(2).sum().detach() + (grad - grad.detach())

def __smooth_fn(As, Bs, n_iterations=100, lr=1e-3, a=None, b=None, device=None):
    if device is None: device = get_device()
    s = torch.nn.Parameter(torch.ones(Bs[0].weight.shape[-1], device=device))
    optimizer = torch.optim.Adam([s], lr=lr)
    for A in As: A.to(device)
    for B in Bs: B.to(device)
    for i in range(n_iterations):
        optimizer.zero_grad()
        loss = 0
        if len(As[0].weight.shape) > 1:
            for A in As: loss += quantization_loss_for_smooth(A.weight, scale=1/s[:,None]).sum()
        for B in Bs: loss += quantization_loss_for_smooth(B.weight, scale=s).sum()
        loss.backward()
        optimizer.step()
        if i == 0 or (i + 1) % 25 == 0:
            print(f"Iteration {i+1}/{n_iterations}, Loss: {loss.item():.6f}")
    print(s)
    s_ = s[:,None] if len(As[0].weight.shape) > 1 else s
    for A in As: A.weight.data = A.weight.float().mul_(s_).to(A.weight.dtype)
    for B in Bs: B.weight.data = B.weight.float().div_(s).to(B.weight.dtype)
    for A in As: A.cpu()
    for B in Bs: B.cpu()

def _decide_step_size(s, index, chunk_idx, loss_fn, current_loss, init_step_size=0.01, r=1.5):
    step = init_step_size
    init_s = s[index]
    best_loss = current_loss
    best_s = init_s
    for i in range(10):
        s[index] = init_s + step
        # step *= r
        step = init_s * (i + 1)
        loss = loss_fn(chunk_idx)
        if best_loss > loss:
            best_loss = loss
            best_s = s[index]
            break
    s[index] = best_s
    return loss

def decide_step_size(s, index, chunk_idx, loss_fn, current_loss, init_step_size=0.025, r=1.5):
    init_s = s[index]
    loss = _decide_step_size(s, index, chunk_idx, loss_fn, current_loss, init_step_size, r)
    tmp_s, s[index] = s[index], init_s
    loss2 = _decide_step_size(s, index, chunk_idx, loss_fn, current_loss, -init_step_size, r)
    if loss < loss2: loss2, s[index] = loss, tmp_s
    return loss2

@torch.no_grad() 
def smooth_fn_(As, Bs, n_iterations=500, a=None, b=None, device=None, chunk_size=32, step_size=0.01):
    if device is None: device = get_device()
    s = torch.ones(Bs[0].weight.shape[-1], device=device)
    for A in As: A.to(device)
    for B in Bs: B.to(device)
    
    # アニーリングパラメータ
    initial_temp = 0.0
    cooling_rate = 0.995
    
    # sを32要素ずつのチャンクに分割
    num_chunks = (len(s) + chunk_size - 1) // chunk_size
    chunks = [slice(i * chunk_size, min((i + 1) * chunk_size, len(s))) for i in range(num_chunks)]
    
    def compute_loss():
        loss = 0
        if len(As[0].weight.shape) > 1:
            for A in As: loss += quantization_loss_for_smooth(A.weight, scale=1/s[:,None]).reshape(A.weight.shape[0],-1).sum(dim=1)
        for B in Bs: loss += quantization_loss_for_smooth(B.weight, scale=s).reshape(-1, B.weight.shape[-1]).sum(dim=0)
        return loss.reshape(num_chunks, -1).sum(dim=1)
        
    # 各チャンクの初期損失を計算
    losses = compute_loss()
    initial_loss = losses.sum()

    for i in range(n_iterations):
        prev_s = s.clone()
        idx = torch.randint(0, chunk_size, (num_chunks,), device=device) + torch.arange(num_chunks, device=device) * chunk_size
        s[idx] += torch.where(torch.rand((num_chunks,), device=device) > .5, step_size, -step_size)
        new_losses = compute_loss()
        s = torch.where((new_losses < losses)[:,None].expand(-1, chunk_size).reshape(-1), s, prev_s)
        losses = torch.minimum(new_losses, losses)

    print(s)
    print(losses.sum() / initial_loss)

    s_ = s[:,None] if len(As[0].weight.shape) > 1 else s
    for A in As: A.weight.data = A.weight.float().mul_(s_).to(A.weight.dtype)
    for B in Bs: B.weight.data = B.weight.float().div_(s).to(B.weight.dtype)
    for A in As: A.cpu()
    for B in Bs: B.cpu()

@torch.no_grad() 
def smooth_fn(As, Bs, n_iterations=500, a=None, b=None, device=None, chunk_size=32, step_size=0.01):
    smooth_fn_(As, Bs, 100, a, b, device, chunk_size, step_size=step_size * 16)
    smooth_fn_(As, Bs, 100, a, b, device, chunk_size, step_size=step_size * 4)
    smooth_fn_(As, Bs, n_iterations, a, b, device, chunk_size, step_size=step_size)


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
        smooth_mlp(l, a, b)
        smooth_qkv(l, a, b)
        l.cpu()
    # smooth_head(model, a, b)