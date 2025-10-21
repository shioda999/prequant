from ..get_module import *
from .core import *

def compress(model, **kwargs):
    layers = get_layers(model)
    W_list = []
    for l in layers:
        W_list.append(get_down(l).weight)
    W = torch.stack(W_list)
    comp = LoRAStackCompressor.from_weights(W, **kwargs)
    W_rec = comp().cpu()
    mse = torch.mean((W_rec - W) ** 2).item()
    print(f"final mse: {mse:.6e}")
    
    for i, l in enumerate(layers):
        get_down(l).weight.data = W_rec[i].to(W.dtype)