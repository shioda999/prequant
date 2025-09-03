
def normalize(A):
    return A.double() / A.double().pow(2).mean().sqrt()