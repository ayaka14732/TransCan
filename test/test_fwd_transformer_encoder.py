from itertools import accumulate, chain, repeat
import jax.numpy as np
from jax.random import PRNGKey, split, uniform
from operator import itemgetter

from fwd_transformer_encoder import fwd_transformer_encoder

seed = 42
keys = map(itemgetter(0), accumulate(chain((split(PRNGKey(seed)),), repeat(None)), lambda acc, _: split(acc[1])))
rand = lambda *shape: uniform(next(keys), shape)

batch_size = 2
n_heads = 3
width_enc = 4
d_k = 5
d_v = 6
d_ff = 7
d_model = 8

self_attn_q_kernel = rand(d_model, n_heads,d_k)
self_attn_q_bias = rand(n_heads, d_k)

self_attn_k_kernel = rand(d_model, n_heads,d_k)
self_attn_k_bias = rand(n_heads, d_k)

self_attn_v_kernel = rand(d_model, n_heads,d_v)
self_attn_v_bias = rand(n_heads, d_v)

self_attn_ff_kernel = rand(n_heads * d_v, d_model)
self_attn_ff_bias = rand(d_model)

self_attn_layer_norm_scale = rand(d_model)
self_attn_layer_norm_bias = rand(d_model)

ff0_kernel = rand(d_model, d_ff)
ff0_bias = rand(d_ff)

ff1_kernel = rand(d_ff, d_model)
ff1_bias = rand(d_model)

final_layer_norm_scale = rand(d_model)
final_layer_norm_bias = rand(d_model)

src = rand(batch_size, width_enc, d_model)

mask_enc_1d = np.array([
    [1, 1, 1, 0],
    [1, 1, 0, 0],
], dtype=np.bool_)
assert mask_enc_1d.shape[0] == batch_size
assert mask_enc_1d.shape[1] == width_enc
mask_enc = np.einsum('bi,bj->bij', mask_enc_1d, mask_enc_1d)[:, None]

params = {
    'self_attn': {
        'q_proj': {'kernel': self_attn_q_kernel, 'bias': self_attn_q_bias},
        'k_proj': {'kernel': self_attn_k_kernel, 'bias': self_attn_k_bias},
        'v_proj': {'kernel': self_attn_v_kernel, 'bias': self_attn_v_bias},
        'ff': {'kernel': self_attn_ff_kernel, 'bias': self_attn_ff_bias},
    },
    'self_attn_layer_norm': {
        'scale': self_attn_layer_norm_scale,
        'bias': self_attn_layer_norm_bias,
    },
    'ff0': {'kernel': ff0_kernel, 'bias': ff0_bias},
    'ff1': {'kernel': ff1_kernel, 'bias': ff1_bias},
    'final_layer_norm': {
        'scale': final_layer_norm_scale,
        'bias': final_layer_norm_bias,
    },
}

output = fwd_transformer_encoder(params, src, mask_enc)

assert output.shape == (batch_size, width_enc, d_model)
