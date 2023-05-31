import jax.nn as nn
import jax.random as rand
from jaxtyping import Array, Bool as B, Float as F, UInt16 as U16, PyTree, jaxtyped

from ..model.dropout import dropout
from ..model.fwd_layer_norm import fwd_layer_norm
from ..model.fwd_linear import fwd_linear
from ..model.fwd_embedding import fwd_embedding
from ..model.fwd_transformer_encoder import fwd_transformer_encoder
from ..random.wrapper import KeyArray

@jaxtyped
def fwd_transformer_encoder_part(
    params: PyTree,
    src: U16[Array, 'bs src_len'],
    mask_enc: B[Array, 'bs 1 src_len src_len'],
    dropout_key: KeyArray=None
) -> F[Array, 'bs src_len d_model']:
    # params
    encoder_embedding: dict = params['encoder_embedding']  # embedding
    encoder_embed_positions: Array = params['encoder_embed_positions']  # array
    encoder_embed_layer_norm: dict = params['encoder_embed_layer_norm']  # layer norm
    encoder_layers: list = params['encoder_layers']  # list of transformer encoder

    proj0: dict = params['proj0']  # linear
    proj1: dict = params['proj1']  # linear

    if dropout_key is not None:
        num_keys = 2 + 2 + len(encoder_layers)
        keys = iter(rand.split(dropout_key, num=num_keys))

    _, width_enc = src.shape

    # https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/bart/modeling_flax_bart.py#L718-L719
    offset = 2

    # encoder
    src = fwd_embedding(encoder_embedding, src)
    src = src + encoder_embed_positions[offset:width_enc+offset]
    src = fwd_layer_norm(encoder_embed_layer_norm, src)

    if dropout_key is not None:
        src = dropout(next(keys), src)

    for encoder_layer in encoder_layers[:-1]:
        if dropout_key is not None:
            src = fwd_transformer_encoder(encoder_layer, src, mask_enc, dropout_key=next(keys))
        else:
            src = fwd_transformer_encoder(encoder_layer, src, mask_enc)

    src = fwd_linear(proj0, src)
    src = nn.gelu(src)
    if dropout_key is not None:
        src = dropout(next(keys), src)

    src = fwd_linear(proj1, src)
    src = nn.gelu(src)
    if dropout_key is not None:
        src = dropout(next(keys), src)

    if dropout_key is not None:
        src = fwd_transformer_encoder(encoder_layers[-1], src, mask_enc, dropout_key=next(keys))
    else:
        src = fwd_transformer_encoder(encoder_layers[-1], src, mask_enc)

    return src
