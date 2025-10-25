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
    comp = LoRAStackCompressor.from_weights(w, **kwargs)
    # comp = CrossLayerAECompressor.from_weights(W, **kwargs)
    w_rec = comp()
    mse = torch.mean((w_rec - w.to(w_rec.device)) ** 2).item()
    print(f"final mse: {mse:.6e}")

    w_rec = w_rec.reshape(-1, group_sz).add(8).cpu().mul(s).add(min_v).reshape(shape)
    
    for i, l in enumerate(layers):
        # get_gate(l).weight.data = (u_list[i] * w_rec[i] * v[i]).to(dtype)
        w_r = w_rec[i].to(dtype)
        w_r = w_r.gather(dim=-2, index=perm[i].argsort(dim=-2).expand_as(w_r))
        get_gate(l).weight.data = w_r

def permute_sim(w):
    N, D, D2 = w.shape
    w = w.cuda()
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
    return perm.cpu()