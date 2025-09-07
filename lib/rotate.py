from .hadamard import generate_hadamard_matrix
from .get_module import *
from .utils import *
import torch

SZ = 32

def fuse_norm(norm, fcs):
    for fc in fcs:
        setattr(fc, "prev_dtype", fc.weight.dtype)
        fc.weight.data = norm.weight.double() * fc.weight.double()
    setattr(norm, "prev_weight", norm.weight.data.clone())
    norm.weight.data = torch.ones_like(norm.weight)

def defuse_norm(norm, fcs, p=2, sz=SZ):
    # s = norm.prev_weight.reshape(sz, -1).abs().mean(dim=0)[None].expand((sz, -1)).reshape(-1).sqrt()
    s = torch.concat([normalize(fc.weight.data) for fc in fcs]).reshape(-1, fcs[0].weight.shape[-1]).abs().pow(p).mean(dim=0).pow(1/p).pow(0.5)
    for fc in fcs:
        fc.weight.data = fc.weight.data.double().div(s).to(fc.prev_dtype)
        del fc.prev_dtype
    norm.weight.data = norm.weight.data.double().mul(s).to(norm.weight.dtype)
    del norm.prev_weight

def rotate(A, H=None):
    if H is None: H = generate_hadamard_matrix(SZ, A.weight.device)
    A.weight.data = (A.weight.reshape(-1, H.shape[0]).double() @ H).to(A.weight.dtype).reshape(A.weight.shape)

def rotate_r(A, H=None):
    N, M = A.weight.shape
    if H is None: H = generate_hadamard_matrix(SZ, A.weight.device)
    A.weight.data = (H.T[None] @ A.weight.reshape(-1, H.shape[0], M).double()).to(A.weight.dtype).reshape(N, M)

def rotate_embedding(model):
    embed = get_embed(model)
    rotate(embed)

def rotate_qkv(layer):
    norm = get_pre_norm(layer)
    qkv = [get_q(layer), get_k(layer), get_v(layer)]
    fuse_norm(norm, qkv)
    for e in qkv: rotate(e)
    defuse_norm(norm, qkv)

def rotate_vo(layer):
    v, o = get_v(layer), get_o(layer)
    rotate_r(v)
    rotate(o)

def rotate_o(layer):
    o = get_o(layer)
    rotate_r(o)

def rotate_mlp(layer):
    norm = get_post_norm(layer)
    up, gate, down = get_up(layer), get_gate(layer),  get_down(layer)
    fuse_norm(norm, [up, gate])
    rotate(up)
    rotate(gate)
    defuse_norm(norm, [up, gate])
    rotate_r(down)

def rotate_head(model):
    norm, head = get_head_norm(model), get_head(model)
    fuse_norm(norm, [head])
    rotate(head)
    defuse_norm(norm, [head])

class RotLinear(torch.nn.Linear):
    def forward(self, x):
        y = super().forward(x)
        sz = SZ
        H = generate_hadamard_matrix(sz, x.device)
        x = (x.double().reshape(x.shape[0], x.shape[1], -1, sz) @ H.T).reshape(x.shape).to(x.dtype)
        A = self.weight.data
        # A = (A.reshape(A.shape[0], -1, sz).double() @ H).to(A.dtype).reshape(A.shape)
        y2 = torch.nn.functional.linear(x, A)
        # print(y[0,0,:20])
        # print(y2[0,0,:20])
        # input()
        return y2
    
def add_rotate_for_norm(norm):
    class RotNorm(norm.__class__):
        def forward(self, x):
            sz = SZ
            H = generate_hadamard_matrix(sz, x.device)
            x = (x.double().reshape(x.shape[0], x.shape[1], -1, sz) @ H.T).reshape(x.shape).to(x.dtype)
            return super().forward(x)
    norm.__class__ = RotNorm

def add_rotate_post_linear(linear):
    class RotLinear(linear.__class__):
        def forward(self, x):
            x = super().forward(x)
            sz = SZ
            H = generate_hadamard_matrix(sz, x.device)
            x = (x.double().reshape(x.shape[0], x.shape[1], -1, sz) @ H).reshape(x.shape).to(x.dtype)
            return x
    linear.__class__ = RotLinear
    
def apply_rotlinear_qkv(layer):
    qkv = [get_q(layer), get_k(layer), get_v(layer)]
    for e in qkv:
        e.__class__ = RotLinear

@torch.no_grad()
def apply_rotate(model):
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight)

    rotate_embedding(model)
    layers = get_layers(model)
    for l in layers:
        rotate_qkv(l)
        rotate_vo(l)
        rotate_o(l)
        rotate_mlp(l)
    rotate_head(model)
