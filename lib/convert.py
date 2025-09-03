from .get_module import get_layers
from .smooth import apply_smooth
from .rotate import apply_rotate

def convert(model):
    apply_rotate(model)
    apply_smooth(model)
