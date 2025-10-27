# from lib.comp.comp import test

# test()

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from lib.utils import *
from lib.get_module import *

import torch
import math
import matplotlib.pyplot as plt
from typing import List, Tuple

# ---------------------------
# ユーティリティ: state_dict から 2D weight を収集
# ---------------------------
def collect_linear_weights(state_dict: dict, key_substrs=("weight",), layer_filter=lambda k: True):
    keys = []
    weights = []
    for k, v in state_dict.items():
        if not layer_filter(k):
            continue
        if any(sub in k for sub in key_substrs):
            if v.dim() == 2:
                weights.append(v.clone().float())
                keys.append(k)
    return keys, weights

# ---------------------------
# 各重みから SVD を取って U, S, V を返す
# ---------------------------
def svd_for_weights(weights: List[torch.Tensor], device='cpu', trunc_rank=None):
    Us = []
    Ss = []
    Vs = []
    for W in weights:
        Wc = W.to(device)
        U, S, Vh = torch.linalg.svd(Wc, full_matrices=False)
        V = Vh.T
        if trunc_rank is not None:
            U = U[:, :trunc_rank]
            S = S[:trunc_rank]
            V = V[:, :trunc_rank]
        Us.append(U.cpu())
        Ss.append(S.cpu())
        Vs.append(V.cpu())
    return Us, Ss, Vs

# ---------------------------
# 類似度指標
# ---------------------------
def principal_angle_singular_values(U1: torch.Tensor, U2: torch.Tensor) -> torch.Tensor:
    # U1: (d, k1), U2: (d, k2). 内積行列は k1 x k2 -> svdvals を返す
    M = U1.T @ U2  # (k1, k2)
    s = torch.linalg.svdvals(M)
    return s

def subspace_similarity_mean(U1: torch.Tensor, U2: torch.Tensor) -> float:
    s = principal_angle_singular_values(U1, U2)
    return float(s.mean().item())

def subspace_similarity_top(U1: torch.Tensor, U2: torch.Tensor) -> float:
    s = principal_angle_singular_values(U1, U2)
    return float(s[0].item())

def frobenius_inner_normalized(A: torch.Tensor, B: torch.Tensor) -> float:
    # ⟨A, B⟩ / (||A||_F ||B||_F)
    num = (A * B).sum()
    den = torch.norm(A) * torch.norm(B) + 1e-12
    return float((num / den).item())

def orthogonal_procrustes_distance(A: torch.Tensor, B: torch.Tensor) -> float:
    # argmin_R ||A R - B||_F  s.t. R^T R = I
    # R = UV^T where U, _, Vt = svd(A^T B)
    # distance = ||A R - B||_F
    M = A.T @ B  # (kA, kB)  ここは通常 kA == kB を仮定
    U, _, Vt = torch.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    AR = A @ R
    dist = torch.norm(AR - B)
    return float(dist.item())

# ---------------------------
# ペアワイズ行列を作る
# ---------------------------
def pairwise_similarity_matrix(matrices: List[torch.Tensor], metric: str = "subspace_mean") -> torch.Tensor:
    n = len(matrices)
    M = torch.zeros((n, n), dtype=torch.float32)
    for i in range(n):
        for j in range(n):
            if metric == "subspace_mean":
                M[i, j] = subspace_similarity_mean(matrices[i], matrices[j])
            elif metric == "subspace_top":
                M[i, j] = subspace_similarity_top(matrices[i], matrices[j])
            elif metric == "frobenius":
                M[i, j] = frobenius_inner_normalized(matrices[i], matrices[j])
            elif metric == "procrustes":
                # procrustes distance is a distance (lower better); convert to similarity by negating if desired
                M[i, j] = orthogonal_procrustes_distance(matrices[i], matrices[j])
            else:
                raise ValueError("unknown metric")
    return M

# ---------------------------
# ヒートマップ描画
# ---------------------------
def plot_heatmap(mat: torch.Tensor, labels: List[str], title: str, vmax=None, vmin=None, savepath=None):
    plt.figure(figsize=(8, 6))
    arr = mat.numpy()
    im = plt.imshow(arr, aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(im)
    plt.title(title)
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=8)
    plt.yticks(range(len(labels)), labels, fontsize=8)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.show()

# ---------------------------
# フルワークフローの関数
# ---------------------------
def analyze_uv_similarity(state_dict: dict,
                          key_substrs=("weight",),
                          layer_filter=lambda k: True,
                          trunc_rank=64,
                          device='cpu',
                          metrics=("subspace_mean", "subspace_top", "frobenius", "procrustes"),
                          plot=True):
    keys, weights = collect_linear_weights(state_dict, key_substrs=key_substrs, layer_filter=layer_filter)
    print(f"found {len(weights)} 2D weights")
    Us, Ss, Vs = svd_for_weights(weights, device=device, trunc_rank=trunc_rank)

    # labels (短くする)
    labels = [k.replace("weight", "").replace(".", "_") for k in keys]
    
    def cosine(a, b):
        return torch.dot(a, b) / (a.norm() * b.norm())
    L = len(Ss)
    sim = torch.zeros(L, L)
    for i in range(L):
        for j in range(L):
            sim[i, j] = cosine(Ss[i], Ss[j])
    plot_heatmap(sim.detach().cpu(), labels, "S")
    exit(0)

    results = {}
    for metric in metrics:
        print(f"computing metric: {metric}")
        M_u = pairwise_similarity_matrix(Us, metric=metric)
        M_v = pairwise_similarity_matrix(Vs, metric=metric)
        results[f"U_{metric}"] = M_u
        results[f"V_{metric}"] = M_v
        if plot:
            title_u = f"U similarity ({metric})"
            title_v = f"V similarity ({metric})"
            # 値域の扱い: procrustes は距離なので逆に表示することも検討
            if metric == "procrustes":
                # smaller = more similar -> invert for visualization
                maxd = float(max(M_u.max(), M_v.max()))
                plot_heatmap(-M_u, labels, title_u + " (negated for similarity)")
                plot_heatmap(-M_v, labels, title_v + " (negated for similarity)")
            else:
                plot_heatmap(M_u, labels, title_u)
                plot_heatmap(M_v, labels, title_v)

    return results, labels

# ---------------------------
# 使い方例（ダミー）
# ---------------------------
# if __name__ == "__main__":
#     # 例: ダミー state_dict を作って試す
#     state = {}
#     for i in range(12):
#         # 適当に変化する行列
#         base = torch.randn(512, 512)
#         perturb = (i * 0.02) * torch.randn_like(base)
#         state[f"layer.{i}.linear.weight"] = base + perturb

#     res, labels = analyze_uv_similarity(state, trunc_rank=64, device='cpu', plot=True)
#     # res["U_subspace_mean"] 等に類似度行列が入る
def cosine(a, b):
    return torch.dot(a, b) / (a.norm() * b.norm())

def get_model(model_name):
    kwargs = { "torch_dtype": torch.float16, "device_map": "cpu" }
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    divide(model)
    apply_config(model)
    model.seqlen = 4096
    return model, tokenizer

def main():
    model, tokenizer = get_model('Qwen/Qwen3-0.6B')
    layers = get_layers(model)
    w_list = []
    state = {}
    for i, l in enumerate(layers):
        w_list.append(get_gate(l).weight.clone())
        state[f"gate.{i}.weight"] = w_list[-1]
    
    # for w in w_list:
    #     U, S, Vh = torch.linalg.svd(w.float(), full_matrices=False)
    #     print(U)
    res, labels = analyze_uv_similarity(state, trunc_rank=64, device='cpu', plot=True)

if __name__ == "__main__":
    main()