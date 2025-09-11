def get_layers(model):
    return model.model.layers

def get_attn(layer):
    return layer.self_attn

def get_q(layer):
    return get_attn(layer).q_proj

def get_k(layer):
    return get_attn(layer).k_proj

def get_v(layer):
    return get_attn(layer).v_proj

def get_o(layer):
    return get_attn(layer).o_proj

def get_pre_norm(layer):
    return layer.input_layernorm

def get_post_norm(layer):
    if hasattr(layer, "pre_feedforward_layernorm"):
        return layer.pre_feedforward_layernorm
    return layer.post_attention_layernorm

def get_up(layer):
    return layer.mlp.up_proj

def get_gate(layer):
    return layer.mlp.gate_proj

def get_down(layer):
    return layer.mlp.down_proj

def get_embed(model):
    return model.model.embed_tokens

def get_head_norm(model):
    return model.model.norm

def get_head(model):
    return model.lm_head

def get_head_dim(layer):
    return get_attn(layer).q_norm.weight.shape[0]