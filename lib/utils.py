
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

@torch.no_grad()
def standarize(model):
    divide(model)
    convert_norm(model)

@torch.no_grad()
def unstandarize(model):
    undivide(model)
    recover_norm(model)

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
            w_q = attn.q_proj.weight
            o_dim, i_dim = w_q.shape
            attn.qkv_proj = torch.nn.Linear(i_dim, head_dim * (num_heads + num_kv_heads * 2),
                                         False, w_q.device, w_q.dtype)
            attn.qkv_proj.weight.data = torch.concat([
                attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight
            ])

            del attn.q_proj, attn.k_proj, attn.v_proj

        if hasattr(model, "gate_up_merge") and model.gate_up_merge:
            w_gate = mlp.gate_proj.weight
            o_dim, i_dim = w_gate.shape
            mlp.gate_up_proj = torch.nn.Linear(i_dim, o_dim * 2, False, w_gate.device, w_gate.dtype)
            mlp.gate_up_proj.weight.data = torch.concat([
                mlp.gate_proj.weight,
                mlp.up_proj.weight,
            ])

            del mlp.gate_proj, mlp.up_proj
    if hasattr(model, "qkv_merge"): del model.qkv_merge
    if hasattr(model, "gate_up_merge"): model.gate_up_merge

class ConcatModule:
    def __init__(self, *modules):
        self.modules = modules
    def __call__(self, x):
        y = torch.concat([m(x) for m in self.modules], dim=-1)
        return y
    
def convert_norm(model):
    if is_gemma_norm(model):
        model._is_gemma = True
        norm_class = get_head_norm(model).__class__
        for m in model.modules():
            if isinstance(m, norm_class):
                m._prev_class = norm_class
                m.__class__ = _RMSNorm
                m.weight += 1

def recover_norm(model):
    if hasattr(model, "_is_gemma"):
        del model._is_gemma
        for m in model.modules():
            if isinstance(m, _RMSNorm):
                m.__class__ = m._prev_class
                del m._prev_class
                m.weight -= 1

def is_gemma_norm(model):
    norm = get_head_norm(model)
    x = torch.zeros((1,norm.weight.shape[0]), device=norm.weight.device)
    x[0] = 1
    return torch.allclose(norm(x), (norm.weight + 1) * x)

class _RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * self.weight.float()
        return output.type_as(x)

@torch.no_grad()
def _quantize(w, nbits=4, group_sz=32):
    shape, dtype = w.shape, w.dtype
    w = w.reshape(-1, group_sz).float()
    Qp, Qn = 2 ** (nbits - 1) - 1, -2 ** (nbits - 1)
    s = torch.maximum(w.max(dim=1, keepdim=True)[0] / Qp, w.min(dim=1, keepdim=True)[0] / Qn)
    w_q = w.div(s).round_().clamp_(Qn, Qp).mul_(s).reshape(shape).to(dtype)
    return w_q, s

def quantize(w, nbits=4, group_sz=32, ste=False):
    shape, dtype = w.shape, w.dtype
    w = w.reshape(-1, group_sz).float()
    Qp = 2 ** nbits - 1
    min_v = w.min(dim=1, keepdim=True)[0]
    s = (w.max(dim=1, keepdim=True)[0] - min_v) / Qp
    round_fn = round_ste if ste else torch.round
    w_q = round_fn(w.sub(min_v).div(s)).clamp(0, Qp).mul(s).add(min_v).reshape(shape).to(dtype)
    return w_q, s

def q_err(m, nbits=4, sz=32, scale=None, act_scale=None, t=False, H=None, o_shrink=True, ste=False, hamiltonian=None):
    w = m.weight if hasattr(m, "weight") else m
    w_q, s = quantize(w, nbits, ste=ste)
    delta = w_q - w
    delta2 = delta
    if scale is not None:
        delta = delta.mul(scale.weight if hasattr(scale, "weight") else scale)
        delta2 = delta
    if hamiltonian is not None and t is False:
        # loss = (delta @ hamiltonian * delta).mean(dim=0)
        loss = (delta.float() @ hamiltonian * delta).sum(dim=-1, keepdim=True).pow(2).mean(dim=0)
        # loss = (delta @ hamiltonian * delta).sum(dim=-1, keepdim=True).pow(2).mean(dim=0)\
        #     + (w.float() @ hamiltonian * delta * 2).mean(dim=0)
    else:
        if act_scale is not None:
            delta = delta.mul(act_scale.to(delta.device))
            delta2.mul_(act_scale.to(delta.device).reshape(-1, sz).pow(2).mean(dim=-1, keepdim=True).sqrt().expand(-1, sz).reshape(act_scale.shape))
            # delta2.mul_(act_scale.to(delta.device).pow(2).mean().sqrt())
        if t is True:
            delta = delta.T
            delta2 = delta2.T
        if H is not None:
            delta = (delta.reshape(-1, H.shape[0]).float() @ H.T).reshape(delta.shape)
            # delta = (delta.reshape(-1, H.shape[0]).float() @ H).reshape(delta.shape)
        delta = delta.float()
        delta2 = delta2.float()
        # loss = delta.float().pow(2).mean(dim=0)
        # loss = delta.pow(2).mean(dim=0) + delta2.pow(2).mean(dim=0)
        loss = delta2.pow(2).mean(dim=0)
        # loss = delta2.pow(2).mean(dim=0)
    if o_shrink:
        loss = loss.reshape(-1, sz).mean(dim=-1)
    return loss

@torch.no_grad()
def calc_quantize_error(model, sz=32, H=None):
    if H is None and hasattr(model, "g_rotate_mat"): H = model.g_rotate_mat
    result = {"!SUM": 0}

    def register(m, labels, nbits=4, norm=None, t=False):
        if norm is None: err = q_err(m, nbits, t=t, sz=sz, H=H)
        else:
            err = q_err(m, nbits, scale=norm.weight, t=t, sz=sz, H=H)
            # act_scale = norm.act_scale.to(norm.weight.device)
            # err = q_err(m, nbits, scale=norm.weight, act_scale=act_scale, t=t, sz=sz, H=H)
            # err = q_err(m, nbits, act_scale=m.act_scale, t=t, sz=sz, H=H)
        result["!SUM"] += err.sum().item()
        for e in labels:
            if e not in result: result[e] = err.sum().item()
            result[e] += err.sum().item()

    register(get_embed(model), ["embed"])
    layers = get_layers(model)
    for i, l in enumerate(layers):
        pre_norm, post_norm = get_pre_norm(l), get_post_norm(l)
        register(get_q(l), ["q", f"{i:02}.q"], norm=pre_norm)
        register(get_k(l), ["k", f"{i:02}.k"], norm=pre_norm)
        register(get_v(l), ["v", f"{i:02}.v"], 6, norm=pre_norm)
        register(get_o(l), ["o", f"{i:02}.o"], t=True)
        register(get_gate(l), ["gate", f"{i:02}.gate"], norm=post_norm)
        register(get_up(l), ["up", f"{i:02}.up"], norm=post_norm)
        register(get_down(l), ["down", f"{i:02}.down"], 6, t=True)
    register(get_head(model), ["head"], norm=get_head_norm(model))

    return result

@torch.no_grad()
def calc_quantize_error_v2(model, sz=32, labels=None, H=None):
    result = {}
    if H is None and hasattr(model, "g_rotate_mat"): H = model.g_rotate_mat

    def register(m, label, nbits=4, norm=None, t=False):
        if labels is None or any([e in label for e in labels]):
            if norm is None: err = q_err(m, nbits, t=t, sz=sz, H=H)
            else:
                act_scale = norm.act_scale.to(norm.weight.device)
                err = q_err(m, nbits, scale=norm.weight, act_scale=act_scale, t=t, sz=sz, H=H)
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

def _sinkhorn(K, r=None, c=None, n_iter=50, eps=1e-9):
    nr, nc = K.shape
    if r is None: r = torch.ones(nr, device=K.device) / nr
    if c is None: c = torch.ones(nc, device=K.device) / nc
    u = torch.ones_like(r)
    v = torch.ones_like(c)
    for _ in range(n_iter):
        u = r / (K @ v + eps)
        v = c / (K.T @ u + eps)
    return u, K, v

def sinkhorn(matrix,
                 order=8,
                 clip_min=1e-3,
                 clip_max=1e3,
                 eps=1e-6,
                 stop_on_increasing_imbalance=True):
    """
    vmap-friendly Sinkhorn that returns *the* mu1 / mu2 corresponding
    to the matrix with the minimal imbalance encountered during the
    iteration.

    The return value is a tuple
        (scaled_matrix, mu1_at_minimum, mu2_at_minimum)
    """
    dtype = torch.float32
    m = matrix.to(dtype)
    dev = m.device
    measure = torch.std

    def imbalance(mat):
        s1, s2 = measure(mat, 1), measure(mat, 0)
        s_min = torch.minimum(s1.min(), s2.min()).clamp_min(1e-12)
        s_max = torch.maximum(s1.max(), s2.max())
        return s_max / s_min          # scalar

    imb_min = torch.tensor(float('inf'), dtype=dtype, device=dev)
    gate    = torch.tensor(0.0, dtype=dtype, device=dev)

    tgt_small = torch.minimum(
        m.std(1).clamp(clip_min, clip_max).min(),
        m.std(0).clamp(clip_min, clip_max).min()
    ) + eps

    log_mu1 = torch.zeros(m.shape[1], dtype=dtype, device=dev)
    log_mu2 = torch.zeros(m.shape[0], 1, dtype=dtype, device=dev)

    # Known-good candidates for the step k=0
    cur0          = m
    ib0           = imbalance(cur0)
    imb_min       = torch.minimum(imb_min, ib0)
    mu1_star      = log_mu1.exp().clone()
    mu2_star      = log_mu2.exp().clone()

    for _ in range(order):
        cur       = (m / log_mu1.exp()) / log_mu2.exp()
        ib        = imbalance(cur)

        # update the best-so-far candidates
        better    = (ib <= imb_min).to(dtype)   # 1 if new best
        imb_min   = torch.min(imb_min, ib)
        mu1_star  = torch.where(better.bool(), log_mu1.exp(), mu1_star)
        mu2_star  = torch.where(better.bool(), log_mu2.exp(), mu2_star)

        # early-exit condition
        if stop_on_increasing_imbalance:
            rising = (ib > imb_min).to(dtype)
            gate   = torch.clip(gate + rising, max=1.0)   # once 1 → always 1

        # still-running samples update the dual variables
        g  = 1.0 - gate

        std_r  = measure(cur, 1).clamp(clip_min, clip_max)
        std_c  = measure(cur,0).clamp(clip_min, clip_max)

        sal_col = (std_c / tgt_small).clamp(0.7, 2.0).log()
        sal_row = (std_r[:, None] / tgt_small).clamp(0.7, 2.0).log()

        log_mu1 = (log_mu1 + (sal_col * g)).clip(-.3, 10.)
        log_mu2 = (log_mu2 + (sal_row * g)).clip(-.3, 10.)

    return mu2_star, m, mu1_star

@torch.no_grad()
def clamp_for_scale(x, ratio=0.01):
    norm = x.norm(dim=-1, keepdim=True)
    thr = norm * ratio
    thr2 = norm / ratio
    return torch.where(x.abs() < thr, thr, torch.where(x.abs() > thr2, thr2, x))