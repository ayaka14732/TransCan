import jax.numpy as np
from jaxtyping import Array, Float as F, PyTree, jaxtyped
from typeguard import check_type, typechecked

@jaxtyped
@typechecked
def fwd_layer_norm(params: PyTree, x: F[Array, '*dims last_dim'], eps: float=1e-5) -> F[Array, '*dims last_dim']:
    # params
    scale: Array = params['scale']  # array
    bias: Array = params['bias']  # array

    check_type('scale', scale, F[Array, 'last_dim'])
    check_type('bias', bias, F[Array, 'last_dim'])

    mean = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)

    check_type('mean', mean, F[Array, '*dims 1'])
    check_type('var', var, F[Array, '*dims 1'])

    y = ((x - mean) / np.sqrt(var + eps)) * scale + bias
    check_type('y', y, F[Array, '*dims last_dim'])

    return y
