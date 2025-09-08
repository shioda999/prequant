
def normalize(A):
    return A.float() / A.float().pow(2).mean().sqrt()