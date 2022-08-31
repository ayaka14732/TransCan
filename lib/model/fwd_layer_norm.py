import jax.numpy as np
from jaxtyping import f as F, PyTree, jaxtyped
from typeguard import check_type, typechecked as typechecker

@jaxtyped
@typechecker
def fwd_layer_norm(params: PyTree, x: F['*dims last_dim'], eps: float=1e-5) -> F['*dims last_dim']:
    # params
    scale: np.ndarray = params['scale']  # array
    bias: np.ndarray = params['bias']  # array

    check_type('scale', scale, F['last_dim'])
    check_type('bias', bias, F['last_dim'])

    mean = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)

    check_type('mean', mean, F['*dims 1'])
    check_type('var', var, F['*dims 1'])

    y = ((x - mean) / np.sqrt(var + eps)) * scale + bias
    check_type('y', y, F['*dims last_dim'])

    return y
