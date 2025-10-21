from ..get_module import *
from .core import *

def compress(model, steps=1000):
    layers = get_layers(model)
    W_list = []
    for l in layers:
        W_list.append(get_down(l).weight)
    W = torch.stack(W_list)
    comp = LoRAStackCompressor.from_weights(W, steps=steps)
    W_rec = comp()
    mse = torch.mean((W_rec.cpu() - W_stack) ** 2).item()
    print(f"final mse: {mse:.6e}")
    
    for i, l in enumerate(layers):
        get_down(l).weight.data = W_rec[i]