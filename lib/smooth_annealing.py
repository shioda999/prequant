import torch
from .get_module import *
from .utils import *
import math
import random

def quantization_loss(w, group_sz=32, nbits=4, scale=None):
    shape = w.shape
    w = (w / scale).reshape(-1, group_sz)
    Qp, Qn = 2 ** (nbits - 1) - 1, -2 ** (nbits - 1)
    s = torch.maximum(w.max(dim=1, keepdim=True)[0] / Qp, w.min(dim=1, keepdim=True)[0] / Qn)#.detach()
    w_q = torch.round(w.div(s)).clamp(Qn, Qp).mul(s)
    delta = w - w_q
    delta = delta.reshape(shape).mul(scale)
    return delta.pow(2).sum().detach()

@torch.no_grad() 
def _smooth_fn(As, Bs, n_iterations=100, a=None, b=None, device=None, chunk_size=32):
    if device is None: device = get_device()
    s = torch.ones(Bs[0].weight.shape[-1], device=device)
    for A in As: A.to(device)
    for B in Bs: B.to(device)
    
    # アニーリングパラメータ
    step_size = 0.1
    initial_temp = 0.0
    cooling_rate = 0.995
    
    # sを32要素ずつのチャンクに分割
    num_chunks = (len(s) + chunk_size - 1) // chunk_size
    chunks = [slice(i * chunk_size, min((i + 1) * chunk_size, len(s))) for i in range(num_chunks)]
    
    # 各チャンクの状態を管理
    chunk_temps = torch.full((num_chunks,), initial_temp, device=device)
    chunk_best_s = [s[chunk_slice].clone() for chunk_slice in chunks]
    
    def compute_loss():
        loss = 0
        if len(As[0].weight.shape) > 1:
            for A in As: loss += quantization_loss(A.weight, scale=1/s[:,None])
        for B in Bs: loss += quantization_loss(B.weight, scale=s)
        return loss
    
    def compute_chunk_loss(chunk_idx):
        """指定されたチャンクの損失を計算"""
        loss = 0
        chunk_slice = chunks[chunk_idx]
        chunk_s = s[chunk_slice]
        
        if len(As[0].weight.shape) > 1:
            for A in As: 
                chunk_weight = A.weight[chunk_slice,:]
                loss += quantization_loss(chunk_weight, scale=1/chunk_s[:,None])
        
        for B in Bs:
            chunk_weight = B.weight[:,chunk_slice]
            loss += quantization_loss(chunk_weight, scale=chunk_s)
        
        return loss
    
    # 各チャンクの初期損失を計算
    chunk_losses = [compute_chunk_loss(i) for i in range(num_chunks)]
    
    for i in range(n_iterations):
        # 各チャンクで並列に試行
        for chunk_idx, chunk_slice in enumerate(chunks):
            chunk_len = chunk_slice.stop - chunk_slice.start
            if chunk_len == 0:
                continue
                
            # このチャンク内でランダムなインデックスを選択
            local_idx = random.randint(0, chunk_len - 1)
            global_idx = chunk_slice.start + local_idx
            
            # 元の値を保存
            old_val = s[global_idx].item()
            
            # ±stepを適用
            step = step_size if random.random() > 0.5 else -step_size
            s[global_idx] += step
            
            # このチャンクの新しい損失を計算
            new_chunk_loss = compute_chunk_loss(chunk_idx)
            
            # メトロポリス基準で受容判定
            accept = False
            if new_chunk_loss < chunk_losses[chunk_idx]:
                accept = True
            else:
                delta = new_chunk_loss - chunk_losses[chunk_idx]
                prob = math.exp(-delta / chunk_temps[chunk_idx])
                accept = random.random() < prob
            
            if accept:
                chunk_losses[chunk_idx] = new_chunk_loss
                chunk_best_s[chunk_idx] = s[chunk_slice].clone()
            else:
                # 元に戻す
                s[global_idx] = old_val
        
        # 全チャンクの温度を更新
        chunk_temps *= cooling_rate
        
        if i == 0 or (i + 1) % 25 == 0:
            total_loss = sum(chunk_losses)
            avg_temp = chunk_temps.mean().item()
            print(f"Iteration {i+1}/{n_iterations}, Loss: {total_loss:.6f}, Temp: {avg_temp:.4f}")
    
    # 最良解を適用
    for chunk_idx, chunk_slice in enumerate(chunks):
        s[chunk_slice] = chunk_best_s[chunk_idx]
    
    print(s, compute_loss())
    s_ = s[:,None] if len(As[0].weight.shape) > 1 else s
    for A in As: A.weight.data = A.weight.float().mul_(s_).to(A.weight.dtype)
    for B in Bs: B.weight.data = B.weight.float().div_(s).to(B.weight.dtype)
    for A in As: A.cpu()
    for B in Bs: B.cpu()

def smooth_fn(As, Bs, n_iterations=100, a=None, b=None, device=None, chunk_size=32):
    # temp
    pass

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
    smooth_head(model, a, b)