from .get_module import get_layers
from .smooth import apply_smooth
from .rotate import apply_rotate
from .utils import apply_config
from .permute import apply_permute, apply_global_permute

def convert(model):
    apply_config(model)
    apply_global_permute(model)
    apply_permute(model, m=1)
    apply_rotate(model)
    apply_smooth(model)
