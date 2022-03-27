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

    q = fwd_linear(q_proj, dst)
    k = fwd_linear(k_proj, src)
    v = fwd_linear(v_proj, src)

    qk = np.einsum('bkhm,bvhm->bhkv', q, k)
    qk = qk / np.sqrt(d_k)
    qk = np.where(mask, qk, np.NINF)
    qk = nn.softmax(qk)
    qk = np.where(mask, qk, 0)

    t = np.einsum('bhkv,bvhm->bkhm', qk, v)
    d0, d1, d2, d3 = t.shape
    t = t.reshape(d0, d1, d2 * d3)

    t = fwd_linear(ff, t)
    return t
