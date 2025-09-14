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
        if model.qkv_merge and hasattr(attn, "q_proj"):
            del attn.qkv_proj
            w_q = attn.q_proj.weight
            o_dim, i_dim = w_q.shape
            attn.qkv_proj = torch.nn.Linear(i_dim, head_dim * (num_heads + num_kv_heads * 2),
                                         False, w_q.device, w_q.dtype)
            attn.qkv_proj.weight.data = torch.concat([
                attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight
            ])

            del attn.q_proj, attn.k_proj, attn.v_proj

        if model.gate_up_merge and hasattr(mlp, "gate_proj"):
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

    