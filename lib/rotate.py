from .hadamard import generate_hadamard_matrix
from .get_module import *
from .utils import *
import torch
import gc

@torch.no_grad()
def fuse_norm(norm, fcs):
    for fc in fcs:
        setattr(fc, "prev_dtype", fc.weight.dtype)
        fc.weight.copy_(norm.weight.float() * fc.weight.float())
    setattr(norm, "prev_weight", norm.weight.data.clone())
    norm.weight.copy_(torch.ones_like(norm.weight))

@torch.no_grad()
def defuse_norm(norm, fcs, p=2):
    # s = norm.prev_weight.reshape(sz, -1).abs().mean(dim=0)[None].expand((sz, -1)).reshape(-1).sqrt()
    s = torch.concat([normalize(fc.weight.data) for fc in fcs]).reshape(-1, fcs[0].weight.shape[-1]).abs().pow(p).mean(dim=0).pow(1/p).pow(0.5)
    for fc in fcs:
        fc.weight.copy_(fc.weight.data.float().div(s).to(fc.prev_dtype))
        del fc.prev_dtype
    norm.weight.copy_(norm.weight.data.float().mul(s).to(norm.weight.dtype))
    del norm.prev_weight

@torch.no_grad()
def rotate(A, H):
    A.weight.copy_((A.weight.reshape(-1, H.shape[0]).float() @ H).to(A.weight.dtype).reshape(A.weight.shape))

@torch.no_grad()
def rotate_r(A, H):
    N, M = A.weight.shape
    A.weight.copy_((H.T[None] @ A.weight.reshape(-1, H.shape[0], M).float()).to(A.weight.dtype).reshape(N, M))

def rotate_embedding(model, H):
    embed = get_embed(model)
    rotate(embed, H)

def rotate_qkv(layer, H):
    norm = get_pre_norm(layer)
    qkv = [get_q(layer), get_k(layer), get_v(layer)]
    fuse_norm(norm, qkv)
    for e in qkv: rotate(e, H)
    defuse_norm(norm, qkv)

def rotate_vo(layer, H):
    v, o = get_v(layer), get_o(layer)
    rotate_r(v, H)
    rotate(o, H)

@torch.no_grad()
def rotate_vo_svd(layer):
    head_dim = get_head_dim(layer)
    v, o = get_v(layer), get_o(layer)
    w_o, w_v = o.weight.data, v.weight.data
    ratio = w_o.shape[0] // w_v.shape[0]

    w_o = w_o.reshape(w_o.shape[0],-1,head_dim).transpose(0,1)
    w_v = w_v.reshape(-1,head_dim,w_v.shape[-1])
    w_v_ex = w_v[None].expand(ratio,-1,-1,-1).transpose(0,1).reshape(-1,head_dim,w_v.shape[-1])

    w_o_sh = w_o.reshape(-1,ratio,w_o.shape[1],head_dim).mean(dim=1)
    B = w_o_sh.float() @ w_v.float()
    U, S, Vh = [], [], []
    for e in B:
        u, s, vh = torch.linalg.svd(e, full_matrices=True)
        U.append(u)
        S.append(s)
        Vh.append(vh)
    U, S, Vh = torch.stack(U), torch.stack(S), torch.stack(Vh)
    # print(U.shape, S.shape, Vh.shape)
    U, S, Vh = U[:,:,:head_dim], S[:,:head_dim], Vh[:,:head_dim]
    # print(U.shape, S.shape, Vh.shape)
    w_v_nx = Vh * S.sqrt()[:,:,None]
    # print(w_v_nx.shape)
    w_v_nx_inv = Vh.transpose(-1,-2) / S.sqrt()[:,None]
    A = w_o.float() @ w_v_ex.float()
    w_o_nx = A @ w_v_nx_inv[None].expand(2,-1,-1,-1).transpose(0,1).reshape(-1,*w_v_nx_inv.shape[1:])
    # print(w_o_nx.shape)
    v.weight.copy_(w_v_nx.reshape(-1,w_v_nx.shape[2]).to(w_v.dtype))
    o.weight.copy_(w_o_nx.transpose(0,1).reshape(w_o_nx.shape[1],-1).to(w_o.dtype))

def rotate_o(layer, H):
    o = get_o(layer)
    rotate_r(o, H)

def rotate_mlp(layer, H):
    norm = get_post_norm(layer)
    up, gate, down = get_up(layer), get_gate(layer),  get_down(layer)
    fuse_norm(norm, [up, gate])
    rotate(up, H)
    rotate(gate, H)
    defuse_norm(norm, [up, gate])
    rotate_r(down, H)

def rotate_head(model, H):
    norm, head = get_head_norm(model), get_head(model)
    fuse_norm(norm, [head])
    rotate(head, H)
    defuse_norm(norm, [head])

@torch.no_grad()
def apply_rotate(model, sz=32):
    H = generate_hadamard_matrix(sz, model.lm_head.weight.device)
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight)

    rotate_embedding(model, H)
    layers = get_layers(model)
    for l in layers:
        rotate_o(l, H)
        rotate_vo(l, H)
        # rotate_vo_svd(l)
        rotate_qkv(l, H)
        rotate_mlp(l, H)
    rotate_head(model, H)
