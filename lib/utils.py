import torch

@torch.no_grad()
def normalize(A):
    return A.float().div(A.float().pow(2).mean().sqrt())