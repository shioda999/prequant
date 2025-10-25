from ..get_module import *
from ..utils import *
from .lora_stack import *
from .ae import *

def compress(model, nbits=4, group_sz=32, **kwargs):
    layers = get_layers(model)
    W_list = []
    u_list, v_list = [], []
    for l in layers:
        w = get_gate(l).weight
        dtype = w.dtype
        W_list.append(w)
        # u, K, v = sinkhorn(w)
        # W_list.append(K)
        # u_list.append(u)
        # v_list.append(v)
    w = torch.stack(W_list)
    shape = w.shape
    with torch.no_grad():
        w = w.reshape(-1, group_sz).float()
        Qp = 2 ** nbits - 1
        min_v = w.min(dim=1, keepdim=True)[0]
        s = (w.max(dim=1, keepdim=True)[0] - min_v) / Qp
        w = torch.round(w.sub(min_v).div(s)).clamp(0, Qp).reshape(shape) - 8
        perm = permute_sim(w)
        w = w.gather(dim=-2, index=perm.expand_as(w))
        w[1:] = w[1:] - w[0:1]
        w = (w + 24) % 16 - 8
    comp = LoRAStackCompressor.from_weights(w, **kwargs)
    w_rec = comp()
    # w_rec = w.clone()
    mse = torch.mean((w_rec - w.to(w_rec.device)) ** 2).item()
    print(f"final mse: {mse:.6e}")

    w_rec[1:] = w_rec[1:] + w_rec[0:1]
    w_rec = (w_rec + 24) % 16 - 8
    w_rec = w_rec.gather(dim=-2, index=perm.to(w_rec.device).argsort(dim=-2).expand_as(w_rec))
    w_rec = w_rec.reshape(-1, group_sz).add(8).cpu().mul(s).add(min_v).reshape(shape)

    for i, l in enumerate(layers):
        # get_gate(l).weight.data = (u_list[i] * w_rec[i] * v[i]).to(dtype)
        w_r = w_rec[i].to(dtype)
        get_gate(l).weight.data = w_r

def permute_sim(w):
    N, D, D2 = w.shape
    w = w.to(get_device()).abs()
    base = w[0:1] # (1, D, D2)
    sim = w @ base.transpose(-1,-2) # (N, D, D)
    perm = torch.zeros((N,D,1), device=w.device).int()
    for i in range(N):
        print(f"{i}/{N}")
        for j in range(D):
            idx = sim[i].argmax().item()
            row, col = idx // D, idx % D
            sim[i,row,:] = -torch.inf
            sim[i,:,col] = -torch.inf
            perm[i,col,0] = row
    return perm.to(torch.int64).cpu()

def test():
    w = (torch.randn((2, 3, 5)) * 16).round()
    w = (w + 168) % 16 - 8
    w_ori = w.clone()
    perm = permute_sim(w)
    w = w.gather(dim=-2, index=perm.expand_as(w).to(torch.int64))
    w[1:] = w[1:] - w[0:1]
    w = (w + 24) % 16 - 8

    w[1:] = w[1:] + w[0:1]
    w = (w + 24) % 16 - 8
    w = w.gather(dim=-2, index=perm.argsort(dim=-2).expand_as(w).to(torch.int64))

    print(w_ori)
    print(w)
    print(torch.allclose(w_ori, w))