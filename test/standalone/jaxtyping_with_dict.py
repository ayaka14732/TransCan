import jax; jax.config.update('jax_platforms', 'cpu')
import jax.numpy as np
from jaxtyping import Array, Float as F, jaxtyped
from typing import TypedDict

# the model contains two parameters `a` and `b`
class ParamType(TypedDict):
    a: F[Array, 'm n p']
    b: F[Array, 'm n']

@jaxtyped
def f(x: ParamType) -> ParamType:
    return x

# test 1

params = {
    'a': np.zeros((2, 3, 4)),
    'b': np.zeros((2, 3)),
}
f(params)  # typechecking will be performed here

# test 2

params = {
    'a': np.zeros((2, 3, 4)),
    'b': np.zeros((2, 4)),
}
try:
    f(params)
except TypeError:
    pass
else:
    raise RuntimeError('Expected TypeError')

# test 3

params = {
    'a': np.zeros((2, 3, 4)),
    'c': np.zeros((2, 4)),
}
try:
    f(params)
except TypeError:
    pass
else:
    raise RuntimeError('Expected TypeError')
