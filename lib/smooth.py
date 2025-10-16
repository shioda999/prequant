import torch
from .get_module import *
from .utils import *
import math
import re
import random

@torch.no_grad()
def _smooth_fn(As, Bs, p=2, a=0., b=0.5):
    sa = torch.concat([normalize(A.weight)[..., None] for A in As], dim=-1).reshape(As[0].weight.shape[0], -1).abs().pow(p).mean(dim=1).pow(1/p)
    sb = torch.concat([normalize(B.weight) for B in Bs]).reshape(-1, Bs[0].weight.shape[-1]).abs().pow(p).mean(dim=0).pow(1/p)
    s = sa.pow(-a) * sb.pow(b)
    s_ = s[:,None] if len(As[0].weight.shape) > 1 else s
    for A in As: A.weight.data = A.weight.float().mul_(s_).to(A.weight.dtype)
    for B in Bs: B.weight.data = B.weight.float().div_(s).to(B.weight.dtype)

def __quantization_loss_for_smooth(w, group_sz=32, nbits=4, scale=None):
    shape = w.shape
    w = w.reshape(-1, group_sz)
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
            for A in As: loss += quantization_loss_for_smooth(A.weight * s[:,None], scale=1/s[:,None]).sum()
        for B in Bs: loss += quantization_loss_for_smooth(B.weight / s, scale=s).sum()
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

def quantization_loss_for_smooth(As, Bs, num_chunks, H, s):
    loss = 0
    losses = []
    if hasattr(Bs[0], "act_scale"):
        sa = torch.concat([A.weight[..., None] for A in As], dim=-1).reshape(As[0].weight.shape[0], -1).abs().pow(2).mean(dim=1).pow(0.5)
        for i, B in enumerate(Bs):
            hamiltonian = getattr(B, "H", None)
            loss += q_err(B.weight / s, scale=s, act_scale=B.act_scale, o_shrink=False, H=H, hamiltonian=hamiltonian).reshape(-1, B.weight.shape[-1]).sum(dim=0)
            # losses.append(q_err(B.weight / s, scale=s, act_scale=B.act_scale, o_shrink=False, H=H, hamiltonian=hamiltonian).reshape(-1, B.weight.shape[-1]).sum(dim=0))
    else:
        sa = torch.concat([A.weight[..., None] for A in As], dim=-1).reshape(As[0].weight.shape[0], -1).abs().pow(2).mean(dim=1).pow(0.5)
        # if len(As[0].weight.shape) > 1:
        #     for A in As: loss += q_err(A.weight * s[:,None], scale=1/s[:,None], o_shrink=False, H=H).reshape(A.weight.shape[0],-1).sum(dim=1)
        for B in Bs: loss += q_err(B.weight / s, scale=s, o_shrink=False, H=H).reshape(-1, B.weight.shape[-1]).sum(dim=0)
    # loss = torch.stack(losses).max(dim=0)[0]
    return loss.reshape(num_chunks, -1).sum(dim=1)

@torch.no_grad() 
def smooth_fn_greedy(As, Bs, n_iterations=500, device=None, chunk_size=32, step_size=0.01):
    if device is None: device = get_device()
    s = torch.ones(Bs[0].weight.shape[-1], device=device)
    for A in As: A.to(device)
    for B in Bs: B.to(device)
    H = As[0].rot_mat.to(device) if hasattr(As[0], "rot_mat") else None
    
    # アニーリングパラメータ
    initial_temp = 0.0
    cooling_rate = 0.995
    
    # sを32要素ずつのチャンクに分割
    num_chunks = (len(s) + chunk_size - 1) // chunk_size
    chunks = [slice(i * chunk_size, min((i + 1) * chunk_size, len(s))) for i in range(num_chunks)]
        
    def compute_loss(s):
        return quantization_loss_for_smooth(As, Bs, num_chunks, H, s)
    
    # 各チャンクの初期損失を計算
    losses = compute_loss(s)
    initial_loss = losses.sum()

    for i in range(n_iterations):
        prev_s = s.clone()
        idx = torch.randint(0, chunk_size, (num_chunks,), device=device) + torch.arange(num_chunks, device=device) * chunk_size
        s[idx] += torch.where(torch.rand((num_chunks,), device=device) > .5, step_size, -step_size)
        new_losses = compute_loss(s)
        s = torch.where((new_losses < losses)[:,None].expand(-1, chunk_size).reshape(-1), s, prev_s)
        losses = torch.minimum(new_losses, losses)

    print(s)
    print(losses.sum() / initial_loss)

    s_ = s[:,None] if len(As[0].weight.shape) > 1 else s
    for A in As: A.weight.data = A.weight.float().mul_(s_).to(A.weight.dtype)
    for B in Bs:
        B.weight.data = B.weight.float().div_(s).to(B.weight.dtype)
        if hasattr(B, "act_scale"): B.act_scale.mul_(s)
    for A in As: A.cpu()
    for B in Bs: B.cpu()

@torch.no_grad()
def smooth_fn_pow(As, Bs, device=None, chunk_size=32):
    if device is None: device = get_device()
    for A in As: A.to(device)
    for B in Bs: B.to(device)
    # Bs_scale = [B.weight.float().abs().mean() for B in Bs]
    Bs_scale = [B.weight.float().pow(2).mean().sqrt() for B in Bs]
    for B, s in zip(Bs, Bs_scale): B.weight.div_(s)
    
    dim = Bs[0].weight.shape[-1]
    num_chunks = (dim + chunk_size - 1) // chunk_size
    chunks = [slice(i * chunk_size, min((i + 1) * chunk_size, dim)) for i in range(num_chunks)]
    H = As[0].rot_mat.to(device) if hasattr(As[0], "rot_mat") else None
    
    def compute_loss(s):
        return quantization_loss_for_smooth(As, Bs, num_chunks, H, s)
        
    def calc_minimum_loss(r):
        loss = compute_loss(r.pow(0))
        p = torch.zeros((num_chunks,), device=device)
        for i in torch.arange(0, 1, 0.05):
            new_loss = compute_loss(r.pow(i))
            p = torch.where(new_loss < loss, i, p)
            loss = torch.minimum(new_loss, loss)
        return r.pow(p[:,None].expand(-1, chunk_size).reshape(-1)), loss

    p = 2
    r = 1 / torch.concat([A.weight[..., None] for A in As], dim=-1).reshape(As[0].weight.shape[0], -1).abs().pow(p).mean(dim=1).pow(1/p)
    r2 = torch.concat([B.weight for B in Bs]).reshape(-1, Bs[0].weight.shape[-1]).abs().pow(p).mean(dim=0).pow(1/p)
    
    s, loss = calc_minimum_loss(r)
    s2, loss2 = calc_minimum_loss(r2)
    s = torch.where((loss < loss2)[:,None].expand(-1, chunk_size).reshape(-1), s, s2)
    loss = torch.where(loss < loss2, loss, loss2)

    # if hasattr(As[0], "act_scale"):
    #     s2, loss2 = calc_minimum_loss(As[0].act_scale)
    #     s = torch.where((loss < loss2)[:,None].expand(-1, chunk_size).reshape(-1), s, s2)
    #     loss = torch.where(loss < loss2, loss, loss2)

    if hasattr(Bs[0], "act_scale"):
        s2, loss2 = calc_minimum_loss(1 / Bs[0].act_scale)
        s = torch.where((loss < loss2)[:,None].expand(-1, chunk_size).reshape(-1), s, s2)
        loss = torch.where(loss < loss2, loss, loss2)

    w_b = torch.concat([B.weight for B in Bs]).float()
    shape = w_b.shape
    w_b = w_b.reshape(-1, chunk_size)
    w_b = w_b - w_b.min(dim=1, keepdim=True)[0]
    qs = w_b.max(dim=1, keepdim=True)[0]
    r = w_b.div(qs).reshape(shape).mean(dim=0)
    s2, loss2 = calc_minimum_loss(r)
    s = torch.where((loss < loss2)[:,None].expand(-1, chunk_size).reshape(-1), s, s2)
    loss = torch.where(loss < loss2, loss, loss2)

    print(s)
    s_ = s[:,None] if len(As[0].weight.shape) > 1 else s
    for A in As: A.weight.data = A.weight.float().mul_(s_).to(A.weight.dtype)
    for B in Bs:
        B.weight.data = B.weight.float().div_(s).to(B.weight.dtype)
        if hasattr(B, "act_scale"): B.act_scale.mul_(s)
    for B, s in zip(Bs, Bs_scale): B.weight.mul_(s)
    for A in As: A.cpu()
    for B in Bs: B.cpu()

@torch.no_grad()
def _smooth_fn_pow(As, Bs, a=None, b=None, device=None, chunk_size=32):
    if device is None: device = get_device()
    for A in As: A.to(device)
    for B in Bs: B.to(device)
    
    dim = Bs[0].weight.shape[-1]
    num_chunks = (dim + chunk_size - 1) // chunk_size
    chunks = [slice(i * chunk_size, min((i + 1) * chunk_size, dim)) for i in range(num_chunks)]
    H = As[0].rot_mat.to(device) if hasattr(As[0], "rot_mat") else None
    
    def compute_loss(s):
        return quantization_loss_for_smooth(As, Bs, num_chunks, H, s)
        
    def calc_minimum_loss(r, r2):
        loss = compute_loss(r.pow(0))
        p = torch.zeros((num_chunks,), device=device)
        p2 = torch.zeros((num_chunks,), device=device)
        min_v, max_v = 0.999, 1.001
        for i in torch.arange(0, 1, 0.05):
            for j in torch.arange(0, 1, 0.05):
                new_loss = compute_loss(r.pow(i)*r2.pow(j))
                p = torch.where(new_loss < loss, i, p)
                p2 = torch.where(new_loss < loss, j, p2)
                loss = torch.minimum(new_loss, loss)
        return r.pow(p[:,None].expand(-1, chunk_size).reshape(-1)) * r2.pow(p2[:,None].expand(-1, chunk_size).reshape(-1)), loss

    p = 2
    r = torch.concat([normalize(B.weight) for B in Bs]).reshape(-1, Bs[0].weight.shape[-1]).abs().pow(p).mean(dim=0).pow(1/p)
    r2 = 1 / Bs[0].act_scale

    s, loss = calc_minimum_loss(r, r2)

    print(s)
    s_ = s[:,None] if len(As[0].weight.shape) > 1 else s
    for A in As: A.weight.data = A.weight.float().mul_(s_).to(A.weight.dtype)
    for B in Bs:
        B.weight.data = B.weight.float().div_(s).to(B.weight.dtype)
        if hasattr(B, "act_scale"): B.act_scale.mul_(s)
    for A in As: A.cpu()
    for B in Bs: B.cpu()

@torch.no_grad()
def __smooth_fn_pow(As, Bs, device=None, chunk_size=32):
    if device is None: device = get_device()
    for A in As: A.to(device)
    for B in Bs: B.to(device)
    
    dim = Bs[0].weight.shape[-1]
    num_chunks = (dim + chunk_size - 1) // chunk_size

    def compute_loss(s):
        return quantization_loss_for_smooth(As, Bs, num_chunks, None, s)
    
    w_b = torch.concat([B.weight for B in Bs]).float()
    shape = w_b.shape
    w_b = w_b.reshape(-1, chunk_size)
    w_b = w_b - w_b.min(dim=1, keepdim=True)[0]
    w_b = w_b.reshape(shape)
    Qp = 2 ** 4 - 1
    s = torch.ones((dim,), dtype=torch.float, device=w_b.device)
    loss = torch.ones((chunk_size,), dtype=torch.float, device=w_b.device) * torch.inf
    act_scale = getattr(Bs[0], "act_scale", None)

    for i in range(5):
        w_b = w_b.reshape(-1, chunk_size)
        qs = w_b.max(dim=1, keepdim=True)[0]
        r = w_b.div(qs).reshape(shape).mean(dim=0)
        r_opt = r
        w_b = w_b.reshape(shape)
        for j in range(10):
            r_ = r.pow(0.1 * j)
            loss_ = q_err(w_b / r_, 4, 32, s * r_)#, act_scale)
            r_opt = torch.where((loss_ < loss)[:,None].expand(-1,chunk_size).reshape(-1), r_, r_opt)
            loss = torch.where(loss_ < loss, loss_, loss)
        r = r_opt
        s *= r
        w_b = w_b / r
        print(i, loss)

    print(s)
    s_ = s[:,None] if len(As[0].weight.shape) > 1 else s
    for A in As: A.weight.data = A.weight.float().mul_(s_).to(A.weight.dtype)
    for B in Bs:
        B.weight.data = B.weight.float().div_(s).to(B.weight.dtype)
        if hasattr(B, "act_scale"): B.act_scale.mul_(s)
    for A in As: A.cpu()
    for B in Bs: B.cpu()

@torch.no_grad()
def smooth_fn_sinkhorn(As, Bs, device=None, chunk_size=32):
    if device is None: device = get_device()
    for A in As: A.to(device)
    for B in Bs: B.to(device)
    
    dim = Bs[0].weight.shape[-1]
    num_chunks = (dim + chunk_size - 1) // chunk_size
    chunks = [slice(i * chunk_size, min((i + 1) * chunk_size, dim)) for i in range(num_chunks)]
    H = As[0].rot_mat.to(device) if hasattr(As[0], "rot_mat") else None
    
    w_b = torch.concat([B.weight for B in Bs]).float()
    # if hasattr(Bs[0], "act_scale"): w_b.mul_(Bs[0].act_scale.pow(0.1))

    s = []
    for i in range(dim // chunk_size):
        u, K, v = sinkhorn(w_b[:,i*chunk_size:(i+1)*chunk_size])
        s.append(v)

    s = torch.concat(s)
    print(s)
    s_ = s[:,None] if len(As[0].weight.shape) > 1 else s
    for A in As: A.weight.data = A.weight.float().mul_(s_).to(A.weight.dtype)
    for B in Bs:
        B.weight.data = B.weight.float().div_(s).to(B.weight.dtype)
        if hasattr(B, "act_scale"): B.act_scale.mul_(s)
    for A in As: A.cpu()
    for B in Bs: B.cpu()

@torch.no_grad()
def flip_sign(As, Bs, n_iterations=100, a=None, b=None, device=None, chunk_size=32):
    if device is None: device = get_device()
    s = torch.ones(Bs[0].weight.shape[-1], device=device)
    for A in As: A.to(device)
    for B in Bs: B.to(device)
    H = As[0].rot_mat.to(device) if hasattr(As[0], "rot_mat") else None
    
    # アニーリングパラメータ
    initial_temp = 0.5#0.1
    cooling_rate = 0.995
    
    # sを32要素ずつのチャンクに分割
    num_chunks = (len(s) + chunk_size - 1) // chunk_size
    chunks = [slice(i * chunk_size, min((i + 1) * chunk_size, len(s))) for i in range(num_chunks)]
    
    def compute_loss(s):
        return quantization_loss_for_smooth(As, Bs, num_chunks, H, s)
        
    # 各チャンクの初期損失を計算
    losses = compute_loss(s)
    initial_loss = losses.sum()
    temperature = initial_temp

    for i in range(n_iterations):
        prev_s = s.clone()
        idx = torch.randint(0, chunk_size, (num_chunks,), device=device) + torch.arange(num_chunks, device=device) * chunk_size
        s[idx] *= -1
        new_losses = compute_loss(s)
        cond = torch.logical_or(new_losses < losses, random.random() < (new_losses - losses).div(temperature).exp())
        s = torch.where(cond[:,None].expand(-1, chunk_size).reshape(-1), s, prev_s)
        losses = torch.minimum(new_losses, losses)
        temperature *= cooling_rate
        # print(losses.sum() / initial_loss)

    print(s)
    print(losses.sum() / initial_loss)

    s_ = s[:,None] if len(As[0].weight.shape) > 1 else s
    for A in As: A.weight.data = A.weight.float().mul_(s_).to(A.weight.dtype)
    for B in Bs:
        B.weight.data = B.weight.float().div_(s).to(B.weight.dtype)
        if hasattr(B, "act_scale"): B.act_scale.mul_(s)
    for A in As: A.cpu()
    for B in Bs: B.cpu()

@torch.no_grad() 
def smooth_fn(As, Bs, n_iterations=500, device=None, chunk_size=32, step_size=0.01, mode="pow", **kwargs):
    parts = re.split(r"[.,+]", mode)
    for m in parts:
        if "pow" in m:
            smooth_fn_pow(As, Bs, device, chunk_size)
        if "greedy" in m:
            smooth_fn_greedy(As, Bs, n_iterations, device, chunk_size, step_size=step_size)
        if "flip_sign" in m:
            flip_sign(As, Bs, 300, device, chunk_size)
        if "sinkhorn" in m:
            smooth_fn_sinkhorn(As, Bs, device, chunk_size)
    # smooth_fn_greedy(As, Bs, 100, device, chunk_size, step_size=step_size * 4)
    # smooth_fn_greedy(As, Bs, 100, device, chunk_size, step_size=step_size)
    # smooth_fn_pow(As, Bs, device, chunk_size)

def smooth_qkv(layer, **kwargs):
    norm = get_pre_norm(layer)
    qkv = [get_q(layer), get_k(layer), get_v(layer)]
    smooth_fn([norm], qkv, **kwargs)

@torch.no_grad()
def smooth_vo(layer, a=0.5, b=0.5, **kwargs):
    head_dim = get_head_dim(layer)
    v, o = get_v(layer), get_o(layer)
    device = get_device()
    v.to(device)
    o.to(device)
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

    v.to("cpu")
    o.to("cpu")

def smooth_mlp(layer, up_down=True, **kwargs):
    norm = get_post_norm(layer)
    up, gate, down = get_up(layer), get_gate(layer), get_down(layer)
    smooth_fn([norm], [up, gate], **kwargs)
    if up_down: smooth_fn([up], [down], **kwargs)

def smooth_head(model, **kwargs):
    norm = get_head_norm(model)
    head = get_head(model)
    smooth_fn([norm], [head], **kwargs)

def apply_smooth(model, device=None, vo=True, **kwargs):
    device = get_device()
    model.cpu()
    layers = get_layers(model)
    for i, l in enumerate(layers):
        # if i < len(layers) - 20: continue
        l.to(device)
        smooth_mlp(l, **kwargs)
        smooth_qkv(l, **kwargs)
        if vo: smooth_vo(l)
        l.cpu()
    if get_embed(model).weight is not get_head(model).weight:
        smooth_head(model, **kwargs)