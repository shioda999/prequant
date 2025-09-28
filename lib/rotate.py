from .hadamard import generate_hadamard_matrix
from .get_module import *
from .utils import *
from .smooth import smooth_vo
from .permute import permute_vo, get_perm_v2
import torch
import gc

@torch.no_grad()
def rotate(A, H):
    A.weight.data = (A.weight.reshape(-1, H.shape[0]).float() @ H).to(A.weight.dtype).reshape(A.weight.shape)

@torch.no_grad()
def rotate_r(A, H):
    N, M = A.weight.shape
    A.weight.data = (H.T[None] @ A.weight.reshape(-1, H.shape[0], M).float()).to(A.weight.dtype).reshape(N, M)

def rotate_embedding(model, H):
    embed = get_embed(model)
    rotate(embed, H)

def rotate_qkv(layer, H):
    norm = get_pre_norm(layer)
    qkv = [get_q(layer), get_k(layer), get_v(layer)]
    fuse_norm(norm, qkv)
    for e in qkv: rotate(e, H)
    mean_norm(norm, H)
    defuse_norm(norm, qkv)

def rotate_vo(layer, H):
    v, o = get_v(layer), get_o(layer)
    rotate_r(v, H)
    rotate(o, H)

def rotate_vo_duquant(layer, sz=32):
    smooth_vo(layer)
    head_dim = get_head_dim(layer)
    v, o = get_v(layer), get_o(layer)
    device = v.weight.device
    ratio = o.weight.shape[1] // v.weight.shape[0]
    n_heads = v.weight.shape[0] // head_dim
    Qs = [[random_rotation_matrix(sz, device) for _ in range(head_dim//sz)] for _ in range(n_heads)]
    Qs2 = [[q for _ in range(ratio)] for q in Qs]
    rotate_r(v, torch.block_diag(*[x for row in Qs for x in row]))
    rotate(o, torch.block_diag(*[xx for row in Qs2 for x in row for xx in x]))
    permute_vo(layer, get_perm_v2)
    Qs = [[random_rotation_matrix(sz, device) for _ in range(head_dim//sz)] for _ in range(n_heads)]
    Qs2 = [[q for _ in range(ratio)] for q in Qs]
    rotate_r(v, torch.block_diag(*[x for row in Qs for x in row]))
    rotate(o, torch.block_diag(*[xx for row in Qs2 for x in row for xx in x]))

@torch.no_grad()
def rotate_vo_svd(layer):
    head_dim = get_head_dim(layer)
    v, o = get_v(layer), get_o(layer)
    w_o, w_v = o.weight.data, v.weight.data
    ratio = w_o.shape[1] // w_v.shape[0]

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
    v.weight.data = w_v_nx.reshape(-1,w_v_nx.shape[2]).to(w_v.dtype)
    o.weight.data = w_o_nx.transpose(0,1).reshape(w_o_nx.shape[1],-1).to(w_o.dtype)

def rotate_o(layer, H):
    o = get_o(layer)
    rotate_r(o, H)

def rotate_mlp(layer, H):
    norm = get_post_norm(layer)
    up, gate, down = get_up(layer), get_gate(layer),  get_down(layer)
    fuse_norm(norm, [up, gate])
    rotate(up, H)
    rotate(gate, H)
    mean_norm(norm, H)
    defuse_norm(norm, [up, gate])
    rotate_r(down, H)

def rotate_head(model, H):
    norm, head = get_head_norm(model), get_head(model)
    fuse_norm(norm, [head])
    rotate(head, H)
    mean_norm(norm, H)
    defuse_norm(norm, [head])


def add_rotate_pre(m, H):
    class PreRot(m.__class__):
        def forward(self, x):
            x = (x.float().reshape(-1, H.shape[0]) @ H.T).reshape(x.shape).to(x.dtype)
            return super().forward(x)
    m.__class__ = PreRot

def add_rotate_post(m, H):
    class PostRot(m.__class__):
        def forward(self, x):
            x = super().forward(x)
            x = (x.float().reshape(-1, H.shape[0]) @ H).reshape(x.shape).to(x.dtype)
            return x
    m.__class__ = PostRot
    
@torch.no_grad()
def _apply_rotate(model, sz=32):
    device = next(model.parameters()).device
    model.cpu()
    H = generate_hadamard_matrix(sz, torch.device("cpu"))
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight)

    rotate_embedding(model, H)
    layers = get_layers(model)
    H = H.to(device)
    for l in layers:
        l.to(device)
        rotate_o(l, H)
        # rotate_vo(l, H)
        # rotate_vo_svd(l)
        rotate_vo_duquant(l)
        rotate_mlp(l, H)
        rotate_qkv(l, H)
        torch.cuda.empty_cache()
        l.cpu()
    rotate_head(model, H.cpu())

@torch.no_grad()
def apply_rotate(model, H):
    if isinstance(H, list):
        H = torch.block_diag(*H)
    device = next(model.parameters()).device
    model.cpu()
    dim = get_dim(model)
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight)

    rotate_embedding(model, H.cpu())
    layers = get_layers(model)
    H = H.to(device)
    for l in layers:
        l.to(device)
        rotate_mlp(l, H)
        rotate_o(l, H)
        # rotate_vo_duquant(l)
        rotate_qkv(l, H)
        torch.cuda.empty_cache()
        l.cpu()
    rotate_head(model, H.cpu())

@torch.no_grad()
def apply_rotate_adaptive(model, sz=32, flags=None, device=None):
    if device is None: device = get_device()
    if flags is None:
        H_list = block_diag_hadamard_adaptive(model, sz)
    else:
        H = generate_hadamard_matrix(sz, torch.device("cpu"))
        eye = torch.eye(sz, device=H.device, dtype=H.dtype)
        H_list = [H if f else eye for f in flags]
    H = torch.block_diag(*H_list)
    apply_rotate(model, H)

@torch.no_grad()
def apply_rotate_vo(model, device=None):
    if device is None: device = get_device()
    model.cpu()
    layers = get_layers(model)
    for l in layers:
        l.to(device)
        rotate_vo_duquant(l)
        torch.cuda.empty_cache()
        l.cpu()
    model.to(device)

def block_diag_hadamard_adaptive(model, sz=32):
    emb = get_embed(model)
    tmp = emb.weight.clone()
    before = q_err(emb, sz=sz)

    H = generate_hadamard_matrix(sz, torch.device("cpu"))
    dim = get_dim(model)
    rotate_embedding(model, H)
    after = q_err(emb, sz=sz)
    emb.weight.data = tmp
    flags = before > after * 1.01
    # flags = torch.logical_and(before > after, before2 > after2)
    print(before > after)
    print(flags)
    eye = torch.eye(sz, device=H.device, dtype=H.dtype)
    H_list = [H if f else eye for f in flags]
    return H_list

def block_diag_hadamard_adaptive_v2(model, sz=32):
    before = calc_quantize_error_v2(model, sz=sz)
    H = generate_hadamard_matrix(sz, torch.device("cpu"))
    apply_rotate(model, H)
    after = calc_quantize_error_v2(model, sz=sz)
    ratios = []
    for k in before:
        r = after[k] / before[k]
        ratios.append(r)
    flag = torch.stack(ratios).sqrt().mean(dim=0) < 1
    eye = torch.eye(sz, device=H.device, dtype=H.dtype)
    H_list = [H if f else eye for f in flag]
    del model
    return H_list

def get_vector_dataset(model):
    D = []
    def vec_append(m, t=False):
        w = m.weight.T.clone() if t else m.weight.clone()
        w = w / w.abs().mean()
        D.append(w)
    vec_append(get_embed(model))
    # for l in get_layers(model):
    #     norm = get_pre_norm(l)
    #     qkv = [get_q(l), get_k(l), get_v(l)]
    #     fuse_norm(norm, qkv)
    #     for e in qkv: vec_append(e)
    #     defuse_norm(norm, qkv)
    #     # vec_append(get_o(l), t=True)
    #     norm = get_post_norm(l)
    #     gate_up, down = [get_gate(l), get_up(l)], get_down(l)
    #     fuse_norm(norm, gate_up)
    #     for e in gate_up: vec_append(e)
    #     defuse_norm(norm, gate_up)
    #     # vec_append(down, t=True)
    # norm, head = get_head_norm(model), get_head(model)
    # fuse_norm(norm, [head])
    # vec_append(head)
    # defuse_norm(norm, [head])
    return LargeMatrixDataset(torch.concat(D))

def quantization_loss_for_rotate(w, group_sz=32, nbits=4, use_ste=False):
    shape = w.shape
    w = w.reshape(-1, group_sz)
    Qp, Qn = 2 ** (nbits - 1) - 1, -2 ** (nbits - 1)
    s = torch.maximum(w.max(dim=1, keepdim=True)[0] / Qp, w.min(dim=1, keepdim=True)[0] / Qn).detach()
    round_fn = round_ste if use_ste else torch.round
    w_q = round_fn(w.div(s)).clamp(Qn, Qp).mul(s)
    delta = w - w_q
    return delta.pow(2).sum()

def apply_rotate_optim(model, lr=0.01, num_iterations=1, batch_size=512, device=None):
    from .cayley import SGDG
    if device is None: device = get_device()
    # w = get_embed(model).weight.clone().float().to(device)
    dataset = get_vector_dataset(model)
    test = False
    
    import platform
    num_workers = 0 if platform.system().lower() == 'windows' else 2
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=not test, num_workers=num_workers)
    model.cpu()
    if test: num_iterations = 100
    dim = get_dim(model)
    sz = 32
    # H = torch.eye(dim, device=device, dtype=torch.float)
    # H = torch.nn.Parameter(H)
    # H = torch.stack([generate_hadamard_matrix(sz, torch.device("cpu")) for _ in range(dim//sz)])
    # Hs = [generate_hadamard_matrix(dim, torch.device("cpu"))]
    Hs = block_diag_hadamard_adaptive(model, sz)
    Hs = [torch.nn.Parameter(H.to(device)) for H in Hs]

    loss_fn = quantization_loss_for_rotate
    # loss_fn = QuantizationLoss.apply
    
    # Cayley optimizer
    # optimizer = SGDG([H], lr=lr, stiefel=True)
    optimizer = SGDG(Hs, lr=lr, stiefel=True)
    
    for i in range(num_iterations):
        loss_history = []
        for j, (vec, _) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Compute loss
            # loss = loss_fn(w @ H)
            vec = vec.detach().to(device)
            H = torch.block_diag(*Hs)
            loss = loss_fn(vec @ H) / vec.shape[0]
            loss_history.append(loss)
            
            # Backward pass
            loss.backward()
            
            # Optimization step with Cayley transform
            optimizer.step()
            
            if test: break

        print(f"Iteration {i+1}/{num_iterations}, Loss: {torch.tensor(loss_history).mean().item():.6f}")
            
    model.to(device)
    # print(Hs)
    H = torch.block_diag(*Hs)
    apply_rotate(model, H.cpu())

