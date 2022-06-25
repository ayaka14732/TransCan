import jax.nn as nn
import jax.numpy as np

from .fwd_linear import fwd_linear

def fwd_attention(params: dict, src: np.ndarray, dst: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # params
    q_proj: dict = params['q_proj']  # linear
    k_proj: dict = params['k_proj']  # linear
    v_proj: dict = params['v_proj']  # linear
    ff: dict = params['ff']  # linear

    _, _, d_k = q_proj['kernel'].shape

    q = np.einsum('bdm,mhk->bdhk', dst, q_proj['kernel'])  # bs, dst_len, n_heads, d_k
    k = np.einsum('bsm,mhk->bshk', src, k_proj['kernel'])  # bs, src_len, n_heads, d_k
    v = np.einsum('bsm,mhv->bshv', src, v_proj['kernel'])  # bs, src_len, n_heads, d_v

    if 'bias' in q_proj:
        q += q_proj['bias']  # bs, dst_len, n_heads, d_k
        k += k_proj['bias']  # bs, src_len, n_heads, d_k
        v += v_proj['bias']  # bs, src_len, n_heads, d_v

    qk = np.einsum('bdhk,bshk->bhds', q, k)  # bs, n_heads, dst_len, src_len
    qk = qk / np.sqrt(d_k)
    qk = np.where(mask, qk, np.NINF)
    qk = nn.softmax(qk)
    qk = np.where(mask, qk, 0)

    t = np.einsum('bhds,bshv->bdhv', qk, v)
    d0, d1, d2, d3 = t.shape
    t = t.reshape(d0, d1, d2 * d3)

    t = fwd_linear(ff, t)
    return t
