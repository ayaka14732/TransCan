import jax.numpy as np
from jaxtyping import f as F, PyTree, jaxtyped
from typeguard import check_type, typechecked as typechecker

@jaxtyped
@typechecker
def fwd_linear(params: PyTree, x: F['...']) -> F['...']:
    # params
    kernel: np.ndarray = params['kernel']  # array
    bias: np.ndarray = params['bias']  # array

    check_type('kernel', kernel, F['...'])
    check_type('bias', bias, F['single_dim'])

    # we don't need shape checking because `np.dot` guarantees it
    y = np.dot(x, kernel) + bias
    check_type('y', y, F['...'])

    return y
