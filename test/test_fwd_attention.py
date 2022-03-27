import flax.linen as fnn
from itertools import accumulate, chain, repeat
import jax
import jax.numpy as np
from jax.random import PRNGKey, split, uniform
from operator import itemgetter

from fwd_attention import fwd_attention

# https://github.com/google/jax/issues/9973#issuecomment-1073579382
jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)

seed = 42
keys = map(itemgetter(0), accumulate(chain((split(PRNGKey(seed)),), repeat(None)), lambda acc, _: split(acc[1])))
rand = lambda *shape: uniform(next(keys), shape)

batch_size = 3
max_sent_len = 4
n_heads = 2
d_model = 6
d_k = 7
d_v = 7
d_ff = 9

src = rand(batch_size, max_sent_len, d_model)
dst = rand(batch_size, max_sent_len, d_model)

q_a = rand(n_heads, d_model, d_k)
k_a = rand(n_heads, d_model, d_k)
v_a = rand(n_heads, d_model, d_v)

q_b = rand(n_heads, d_k)
k_b = rand(n_heads, d_k)
v_b = rand(n_heads, d_v)

ff_a = rand(n_heads * d_v, d_ff)
ff_b = rand(d_ff)

mask_dec_1d = np.ones((batch_size, max_sent_len), dtype=np.bool_)
mask = np.einsum('bi,bj->bij', mask_dec_1d, mask_dec_1d)[:, None]

params = {
    'q_proj': {'kernel': q_a, 'bias': q_b},
    'k_proj': {'kernel': k_a, 'bias': k_b},
    'v_proj': {'kernel': v_a, 'bias': v_b},
    'ff': {'kernel': ff_a, 'bias': ff_b},
}
output = fwd_attention(params, src, dst, mask)

# Flax implementation

model = fnn.MultiHeadDotProductAttention(num_heads=n_heads, qkv_features=d_k * n_heads, out_features=d_ff, broadcast_dropout=False)

# params = model.init(jax.random.PRNGKey(0), src, dst)
# jax.tree_map(lambda x: x.shape, params['params'])

output_ = model.apply({'params': {
    'query': {'kernel': q_a.transpose(1, 0, 2), 'bias': q_b},
    'key': {'kernel': k_a.transpose(1, 0, 2), 'bias': k_b},
    'value': {'kernel': v_a.transpose(1, 0, 2), 'bias': v_b},
    'out': {'kernel': ff_a.reshape(n_heads, d_v, d_ff), 'bias': ff_b},
}}, dst, src, mask=mask)

assert np.allclose(output, output_)
