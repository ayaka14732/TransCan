import jax.numpy as np
from jaxtyping import Array, Float as F, PyTree, jaxtyped
from typeguard import check_type, typechecked

@jaxtyped
@typechecked
def fwd_linear(params: PyTree, x: F[Array, '...']) -> F[Array, '...']:
    # params
    kernel: Array = params['kernel']  # array
    bias: Array = params['bias']  # array

    check_type('kernel', kernel, F[Array, '...'])
    check_type('bias', bias, F[Array, 'single_dim'])

    # we don't need shape checking because `np.dot` guarantees it
    y = np.dot(x, kernel) + bias
    check_type('y', y, F[Array, '...'])

    return y
