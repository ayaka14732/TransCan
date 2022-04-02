import jax.numpy as np
import jax.random as rand

from .dropout import dropout
from .fwd_layer_norm import fwd_layer_norm
from .fwd_embedding import fwd_embedding
from .fwd_transformer_encoder import fwd_transformer_encoder
from .fwd_transformer_decoder import fwd_transformer_decoder

def fwd_transformer(params: dict, src: np.ndarray, dst: np.ndarray, mask_enc: np.ndarray, mask_dec: np.ndarray, mask_dec_enc: np.ndarray, dropout_key: rand.KeyArray=None) -> np.ndarray:
    # params
    embedding: dict = params['embedding']  # embedding
    encoder_embed_positions: np.ndarray = params['encoder_embed_positions']  # array
    decoder_embed_positions: np.ndarray = params['decoder_embed_positions']  # array
    encoder_embed_layer_norm: dict = params['encoder_embed_layer_norm']  # layer norm
    decoder_embed_layer_norm: dict = params['decoder_embed_layer_norm']  # layer norm
    encoder_layers: list = params['encoder_layers']  # list of transformer encoder
    decoder_layers: list = params['decoder_layers']  # list of transformer encoder

    if dropout_key is not None:
        num_keys = 2 + len(encoder_layers) + len(decoder_layers)
        keys = iter(rand.split(dropout_key, num=num_keys))

    _, width_enc = src.shape
    _, width_dec = dst.shape

    # https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/bart/modeling_flax_bart.py#L718-L719
    offset = 2

    # encoder
    src = fwd_embedding(embedding, src)
    src = src + encoder_embed_positions[offset:width_enc+offset]
    src = fwd_layer_norm(encoder_embed_layer_norm, src)

    if dropout_key is not None:
        src = dropout(next(keys), src)

    for encoder_layer in encoder_layers:
        if dropout_key is not None:
            src = fwd_transformer_encoder(encoder_layer, src, mask_enc, dropout_key=next(keys))
        else:
            src = fwd_transformer_encoder(encoder_layer, src, mask_enc)

    # decoder
    dst = fwd_embedding(embedding, dst)
    dst = dst + decoder_embed_positions[offset:width_dec+offset]
    dst = fwd_layer_norm(decoder_embed_layer_norm, dst)

    if dropout_key is not None:
        dst = dropout(next(keys), dst)

    for decoder_layer in decoder_layers:
        if dropout_key is not None:
            dst = fwd_transformer_decoder(decoder_layer, src, dst, mask_dec, mask_dec_enc, dropout_key=next(keys))
        else:
            dst = fwd_transformer_decoder(decoder_layer, src, dst, mask_dec, mask_dec_enc)

    return dst
