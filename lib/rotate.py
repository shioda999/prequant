from .hadamard import generate_hadamard_matrix
from .get_module import *
from .utils import *
import torch

def fuse_norm(norm, fcs):
    for fc in fcs:
        setattr(fc, "prev_dtype", fc.weight.dtype)
        fc.weight.data = norm.weight.double() * fc.weight.double()
    norm.weight.data = torch.ones_like(norm.weight)

def defuse_norm(norm, fcs, p=2):
    s = torch.concat([normalize(fc.weight.data) for fc in fcs]).reshape(-1, fcs[0].weight.shape[-1]).abs().pow(p).mean(dim=0).pow(1/p).pow(0.5)
    for fc in fcs:
        fc.weight.data = fc.weight.data.double().div(s).to(fc.prev_dtype)
        del fc.prev_dtype
    norm.weight.data = norm.weight.data.double().mul(s).to(norm.weight.dtype)

def rotate_had(A, sz=1):
    N, M = A.weight.shape
    H = generate_hadamard_matrix(sz, A.weight.device)
    A.weight.data = (A.weight.reshape(N, -1, sz).double() @ H).to(A.weight.dtype).reshape(N, M)

def rotate_had_r(A, sz=1):
    N, M = A.weight.shape
    H = generate_hadamard_matrix(sz, A.weight.device)
    A.weight.data = (H.T[None] @ A.weight.reshape(-1, sz, M).double()).to(A.weight.dtype).reshape(N, M)
    # print(A.weight.shape)
    # A.weight.data = A.weight.data.reshape(N, M)

def rotate_embedding(model):
    embed = get_embed(model)
    rotate_had(embed)

def rotate_qkv(layer):
    norm = get_pre_norm(layer)
    qkv = [get_q(layer), get_k(layer), get_v(layer)]
    fuse_norm(norm, qkv)
    for e in qkv: rotate_had(e)
    defuse_norm(norm, qkv)

def rotate_vo(layer):
    v, o = get_v(layer), get_o(layer)
    rotate_had_r(v)
    rotate_had(o)

def rotate_o(layer):
    o = get_o(layer)
    rotate_had_r(o)

def rotate_mlp(layer):
    norm = get_post_norm(layer)
    up, gate, down = get_up(layer), get_gate(layer),  get_down(layer)
    fuse_norm(norm, [up, gate])
    rotate_had(up)
    rotate_had(gate)
    rotate_had_r(down)
    defuse_norm(norm, [up, gate])

def rotate_head(model):
    norm, head = get_head_norm(model), get_head(model)
    fuse_norm(norm, [head])
    rotate_had(head)
    defuse_norm(norm, [head])

@torch.no_grad()
def apply_rotate(model):
    rotate_embedding(model)
    layers = get_layers(model)
    for l in layers:
        rotate_qkv(l)
        rotate_o(l)
        rotate_mlp(l)
    rotate_head(model)