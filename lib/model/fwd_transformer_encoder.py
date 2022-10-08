import jax.nn as nn
import jax.random as rand
from jaxtyping import Array, Bool as B, Float as F, PyTree, jaxtyped
from typeguard import check_type, typechecked as typechecker

from .dropout import dropout
from .fwd_layer_norm import fwd_layer_norm
from .fwd_linear import fwd_linear
from .fwd_attention import fwd_attention
from ..random.wrapper import KeyArray

@jaxtyped
@typechecker
def fwd_transformer_encoder(
    params: PyTree,
    src: F[Array, 'bs src_len d_model'],
    mask_enc: B[Array, 'bs 1 src_len src_len'],
    dropout_key: KeyArray=None
) -> F[Array, 'bs src_len d_model']:
    # params
    self_attn: dict = params['self_attn']  # attention
    self_attn_layer_norm: dict = params['self_attn_layer_norm']  # layer norm
    ff0: dict = params['ff0']  # linear
    ff1: dict = params['ff1']  # linear
    final_layer_norm: dict = params['final_layer_norm']  # layer norm

    if dropout_key is not None:
        subkeys = rand.split(dropout_key, num=3)

    src_ = src
    t = fwd_attention(self_attn, src, src, mask_enc)
    if dropout_key is not None:
        t = dropout(subkeys[0], t)
    t = t + src_
    t = fwd_layer_norm(self_attn_layer_norm, t)
    check_type('t', t, F[Array, 'bs src_len d_ff'])

    t_ = t
    t = fwd_linear(ff0, t)
    t = nn.gelu(t)
    if dropout_key is not None:
        t = dropout(subkeys[1], t)
    t = fwd_linear(ff1, t)
    if dropout_key is not None:
        t = dropout(subkeys[2], t)
    t = t + t_
    t = fwd_layer_norm(final_layer_norm, t)
    check_type('t', t, F[Array, 'bs src_len d_model'])

    return t
