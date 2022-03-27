import jax.nn as nn
import jax.numpy as np

from .fwd_layer_norm import fwd_layer_norm
from .fwd_linear import fwd_linear
from .fwd_attention import fwd_attention

def fwd_transformer_decoder(params: dict, src: np.ndarray, dst: np.ndarray, mask_dec: np.ndarray, mask_dec_enc: np.ndarray) -> np.ndarray:
    # params
    self_attn: dict = params['self_attn']  # attention
    self_attn_layer_norm: dict = params['self_attn_layer_norm']  # layer norm
    cross_attn: dict = params['cross_attn']  # attention
    cross_attn_layer_norm: dict = params['cross_attn_layer_norm']  # layer norm
    ff0: dict = params['ff0']  # linear
    ff1: dict = params['ff1']  # linear
    final_layer_norm: dict = params['final_layer_norm']  # layer norm

    dst_ = dst
    dst = fwd_attention(self_attn, dst, dst, mask_dec)
    dst = dst + dst_
    dst = fwd_layer_norm(self_attn_layer_norm, dst)

    dst_ = dst
    src = fwd_attention(cross_attn, src, dst, mask_dec_enc)
    t = src + dst_
    t = fwd_layer_norm(cross_attn_layer_norm, t)

    t_ = t
    t = fwd_linear(ff0, t)
    t = nn.gelu(t)
    t = fwd_linear(ff1, t)
    t = t + t_
    t = fwd_layer_norm(final_layer_norm, t)
    return t
