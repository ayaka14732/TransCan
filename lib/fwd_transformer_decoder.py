import jax.nn as nn
import jax.numpy as np
import jax.random as rand

from .dropout import dropout
from .fwd_layer_norm import fwd_layer_norm
from .fwd_linear import fwd_linear
from .fwd_attention import fwd_attention

def fwd_transformer_decoder(params: dict, src: np.ndarray, dst: np.ndarray, mask_dec: np.ndarray, mask_dec_enc: np.ndarray, dropout_key: rand.KeyArray=None) -> np.ndarray:
    # params
    self_attn: dict = params['self_attn']  # attention
    self_attn_layer_norm: dict = params['self_attn_layer_norm']  # layer norm
    cross_attn: dict = params['cross_attn']  # attention
    cross_attn_layer_norm: dict = params['cross_attn_layer_norm']  # layer norm
    ff0: dict = params['ff0']  # linear
    ff1: dict = params['ff1']  # linear
    final_layer_norm: dict = params['final_layer_norm']  # layer norm

    if dropout_key is not None:
        subkeys = rand.split(dropout_key, num=4)

    dst_ = dst
    dst = fwd_attention(self_attn, dst, dst, mask_dec)
    if dropout_key is not None:
        dst = dropout(subkeys[0], dst)
    dst = dst + dst_
    dst = fwd_layer_norm(self_attn_layer_norm, dst)

    dst_ = dst
    src = fwd_attention(cross_attn, src, dst, mask_dec_enc)
    if dropout_key is not None:
        src = dropout(subkeys[1], src)
    t = src + dst_
    t = fwd_layer_norm(cross_attn_layer_norm, t)

    t_ = t
    t = fwd_linear(ff0, t)
    t = nn.gelu(t)
    if dropout_key is not None:
        t = dropout(subkeys[2], t)
    t = fwd_linear(ff1, t)
    if dropout_key is not None:
        t = dropout(subkeys[3], t)
    t = t + t_
    t = fwd_layer_norm(final_layer_norm, t)
    return t
