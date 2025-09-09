import torch

@torch.no_grad()
def normalize(A):
    return A.float().div(A.float().pow(2).mean().sqrt())

def random_rotation_matrix(dim: int, device):
    A = torch.randn(dim, dim)
    Q, R = torch.linalg.qr(A)
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q.to(device)