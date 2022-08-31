import jax.nn as nn
import jax.random as rand
from jaxtyping import b as B, f as F, PyTree, jaxtyped
from typeguard import check_type, typechecked as typechecker

from .dropout import dropout
from .fwd_layer_norm import fwd_layer_norm
from .fwd_linear import fwd_linear
from .fwd_attention import fwd_attention
from ..random.wrapper import KeyArray

@jaxtyped
@typechecker
def fwd_transformer_decoder(
    params: PyTree,
    src: F['bs src_len d_model'],
    dst: F['bs dst_len d_model'],
    mask_dec: B['bs 1 dst_len dst_len'],
    mask_dec_enc: B['bs 1 dst_len src_len'],
    dropout_key: KeyArray=None
) -> F['bs dst_len d_model']:
    # params
    self_attn: PyTree = params['self_attn']  # attention
    self_attn_layer_norm: PyTree = params['self_attn_layer_norm']  # layer norm
    cross_attn: PyTree = params['cross_attn']  # attention
    cross_attn_layer_norm: PyTree = params['cross_attn_layer_norm']  # layer norm
    ff0: PyTree = params['ff0']  # linear
    ff1: PyTree = params['ff1']  # linear
    final_layer_norm: PyTree = params['final_layer_norm']  # layer norm

    if dropout_key is not None:
        subkeys = rand.split(dropout_key, num=4)

    dst_ = dst
    dst = fwd_attention(self_attn, dst, dst, mask_dec)
    if dropout_key is not None:
        dst = dropout(subkeys[0], dst)
    dst = dst + dst_
    dst = fwd_layer_norm(self_attn_layer_norm, dst)
    check_type('dst', dst, F['bs dst_len d_ff'])

    dst_ = dst
    src = fwd_attention(cross_attn, src, dst, mask_dec_enc)
    if dropout_key is not None:
        src = dropout(subkeys[1], src)
    t = src + dst_
    t = fwd_layer_norm(cross_attn_layer_norm, t)
    check_type('t', t, F['bs dst_len d_ff'])

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
    check_type('t', t, F['bs dst_len d_model'])

    return t
