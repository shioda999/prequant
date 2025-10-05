
import torch
import torch.nn.functional as F
from torch.autograd import Function
from .get_module import *
from tqdm import tqdm

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
        if hasattr(model, "qkv_merge") and model.qkv_merge:
            del model.qkv_merge
            w_q = attn.q_proj.weight
            o_dim, i_dim = w_q.shape
            attn.qkv_proj = torch.nn.Linear(i_dim, head_dim * (num_heads + num_kv_heads * 2),
                                         False, w_q.device, w_q.dtype)
            attn.qkv_proj.weight.data = torch.concat([
                attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight
            ])

            del attn.q_proj, attn.k_proj, attn.v_proj

        if hasattr(model, "gate_up_merge") and model.gate_up_merge:
            del model.gate_up_merge
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

@torch.no_grad()
def _quantize(w, nbits=4, group_sz=32):
    shape, dtype = w.shape, w.dtype
    w = w.reshape(-1, group_sz).float()
    Qp, Qn = 2 ** (nbits - 1) - 1, -2 ** (nbits - 1)
    s = torch.maximum(w.max(dim=1, keepdim=True)[0] / Qp, w.min(dim=1, keepdim=True)[0] / Qn)
    w_q = w.div(s).round_().clamp_(Qn, Qp).mul_(s).reshape(shape).to(dtype)
    return w_q, s

@torch.no_grad()
def quantize(w, nbits=4, group_sz=32):
    shape, dtype = w.shape, w.dtype
    w = w.reshape(-1, group_sz).float()
    Qp = 2 ** nbits - 1
    min_v = w.min(dim=1, keepdim=True)[0]
    s = (w.max(dim=1, keepdim=True)[0] - min_v) / Qp
    w_q = w.sub(min_v).div(s).round_().clamp_(0, Qp).mul_(s).add_(min_v).reshape(shape).to(dtype)
    return w_q, s

@torch.no_grad()
def q_err(m, nbits=4, sz=32, scale=None, t=False, H=None, o_shrink=True):
    w = m.weight if hasattr(m, "weight") else m
    w_q, s = quantize(w, nbits)
    delta = w_q - w
    if scale is not None:
        delta.mul_(scale.weight if hasattr(scale, "weight") else scale)
    if t is True:
        delta = delta.T
    if H is not None:
        delta = (delta.reshape(-1, H.shape[0]).float() @ H.T).reshape(delta.shape)
        # delta = (delta.reshape(-1, H.shape[0]).float() @ H).reshape(delta.shape)
    if o_shrink:
        return delta.reshape(delta.shape[0],-1, sz).float().pow(2).mean(dim=-1).mean(dim=0)
    else:
        return delta.float().pow(2).mean(dim=0)

@torch.no_grad()
def calc_quantize_error(model, sz=32, H=None):
    if H is None and hasattr(model, "g_rotate_mat"): H = model.g_rotate_mat
    result = {"!SUM": 0}

    def register(m, labels, nbits=4, norm=None, t=False):
        err = q_err(m, nbits, scale=norm.weight * norm.act_scale, t=t, sz=sz, H=H)
        result["!SUM"] += err.sum().item()
        for e in labels:
            if e not in result: result[e] = err.sum().item()
            result[e] += err.sum().item()

    register(get_embed(model), ["embed"])
    layers = get_layers(model)
    for i, l in enumerate(layers):
        pre_norm, post_norm = get_pre_norm(l), get_post_norm(l)
        register(get_q(l), ["q", f"{i:02}"], norm=pre_norm)
        register(get_k(l), ["k", f"{i:02}"], norm=pre_norm)
        register(get_v(l), ["v", f"{i:02}"], 6, norm=pre_norm)
        register(get_o(l), ["o", f"{i:02}"], t=True)
        register(get_gate(l), ["gate", f"{i:02}"], norm=post_norm)
        register(get_up(l), ["up", f"{i:02}"], norm=post_norm)
        register(get_down(l), ["down", f"{i:02}"], 6, t=True)
    register(get_head(model), ["head"], norm=get_head_norm(model))

    return result

@torch.no_grad()
def calc_quantize_error_v2(model, sz=32, labels=None, H=None):
    result = {}
    if H is None and hasattr(model, "g_rotate_mat"): H = model.g_rotate_mat

    def register(m, label, nbits=4, norm=None, t=False):
        if labels is None or any([e in label for e in labels]):
            err = q_err(m, nbits, scale=norm.weight * norm.act_scale, t=t, sz=sz, H=H)
            result[label] = err

    register(get_embed(model), "embed")
    layers = get_layers(model)
    for i, l in enumerate(layers):
        pre_norm, post_norm = get_pre_norm(l), get_post_norm(l)
        register(get_q(l), f"{i:02}.q", norm=pre_norm)
        register(get_k(l), f"{i:02}.k", norm=pre_norm)
        register(get_v(l), f"{i:02}.v", 6, norm=pre_norm)
        register(get_o(l), f"{i:02}.o", t=True)
        register(get_gate(l), f"{i:02}.gate", norm=post_norm)
        register(get_up(l), f"{i:02}.up", norm=post_norm)
        register(get_down(l), f"{i:02}.down", 6, t=True)
    register(get_head(model), "head", norm=get_head_norm(model))

    return result

@torch.no_grad()
def fuse_norm(norm, fcs):
    for fc in fcs:
        setattr(fc, "prev_dtype", fc.weight.dtype)
        fc.weight.data = norm.weight.float() * fc.weight.float()
    setattr(norm, "prev_weight", norm.weight.data.clone())
    norm.weight.data = torch.ones_like(norm.weight, dtype=norm.weight.dtype)

@torch.no_grad()
def smooth_defuse_norm(norm, fcs, p=2):
    # s = norm.prev_weight.reshape(sz, -1).abs().mean(dim=0)[None].expand((sz, -1)).reshape(-1).sqrt()
    s = torch.concat([normalize(fc.weight.data) for fc in fcs]).reshape(-1, fcs[0].weight.shape[-1]).abs().pow(p).mean(dim=0).pow(1/p).pow(0.5)
    for fc in fcs:
        fc.weight.data = fc.weight.data.float().div(s).to(fc.prev_dtype)
        del fc.prev_dtype
    norm.weight.data = norm.weight.data.float().mul(s).to(norm.weight.dtype)
    del norm.prev_weight

def safe_divide(numerator, denominator, eps=1e-6):
    return numerator / (denominator + (denominator.abs() < eps) * eps)

@torch.no_grad()
def defuse_norm(norm, fcs):
    if hasattr(norm, "smooth_defuse"): return smooth_defuse_norm(norm, fcs)
    for fc in fcs:
        fc.weight.data = safe_divide(fc.weight.float(), norm.prev_weight.float()).to(fc.prev_dtype)
        del fc.prev_dtype
    norm.weight.data = norm.prev_weight.to(norm.weight.dtype)
    del norm.prev_weight
    

@torch.no_grad()
def mean_norm(norm, H):
    t = torch.where((H - torch.eye(H.shape[0], device=H.device)).abs().sum(dim=0)[None].expand(norm.prev_weight.shape[0] // H.shape[0], -1).reshape(-1) == 0, norm.prev_weight, 1.)
    # t = torch.ones_like(norm.prev_weight)
    # t = (norm.prev_weight.abs().float().reshape(-1, H.shape[0]) @ H.abs()).reshape(-1).pow(0.25)# * norm.prev_weight.float().sign()
    norm.prev_weight = t
    # norm.smooth_defuse = True

@torch.no_grad()
def apply_quantize(model):
    for l in get_layers(model):
        for m in [get_q(l), get_k(l), get_o(l),
                   get_gate(l), get_up(l)]:
            m.weight.data, _ = quantize(m.weight)
        for m in [get_v(l), get_down(l)]:
            m.weight.data, _ = quantize(m.weight, 6)
    
    for m in [get_embed(model), get_head(model)]:
        m.weight.data, _ = quantize(m.weight)

@torch.no_grad()
def calc_metric(m, t=False):
    w = m.weight if not t else m.weight.T
    t = w.abs().pow(2).mean(dim=0).sqrt()
    return t / t.mean()

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def round_ste(w):
    return w.round() + w - w.detach()

def quantization_loss(w, group_sz=32, nbits=4, scale=None, use_ste=False):
    shape = w.shape
    w = w.reshape(-1, group_sz)
    Qp, Qn = 2 ** (nbits - 1) - 1, -2 ** (nbits - 1)
    s = torch.maximum(w.max(dim=1, keepdim=True)[0] / Qp, w.min(dim=1, keepdim=True)[0] / Qn)
    round_fn = round_ste if use_ste else torch.round
    w_q = round_fn(w.div(s)).clamp(Qn, Qp).mul(s)
    delta = w_q - w
    if scale is not None: delta = delta.reshape(shape).mul(scale)
    return delta.pow(2).sum()

class LargeMatrixDataset(torch.utils.data.Dataset):
    """
    巨大な行列からミニバッチでベクトルを取得するデータセット
    """
    def __init__(self, matrix, indices=None, transform=None):
        self.matrix = matrix.float().detach()
            
        if indices is None:
            self.indices = torch.arange(self.matrix.size(0))
        else:
            if isinstance(indices, list):
                self.indices = torch.tensor(indices)
            else:
                self.indices = indices
                
        self.transform = transform
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        指定されたインデックスのベクトルを取得
        """
        matrix_idx = self.indices[idx % len(self.indices)]
        vector = self.matrix[matrix_idx]
        
        if self.transform:
            vector = self.transform(vector)
            
        return vector, matrix_idx
    
def grad_change(x, grad):
    return x.detach() + grad - grad.detach()
