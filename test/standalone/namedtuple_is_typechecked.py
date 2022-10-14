import jax; jax.config.update('jax_platforms', 'cpu')
import jax.numpy as jnp
from jaxtyping import Array, Float as F, jaxtyped
from typeguard import typechecked as typechecker
from typing import NamedTuple

class DataCorrect(NamedTuple):
    x: F[Array, 'bs m//2']
    y: F[Array, 'bs m-1']

class DataWrong(NamedTuple):
    x: F[Array, 'bs m//2']
    y: F[Array, 'bs m+1']

@jaxtyped
@typechecker
def correct(a: F[Array, 'bs m']) -> DataCorrect:
    _, m = a.shape
    return DataCorrect(a[:, :m // 2], a[:, :m - 1])

@jaxtyped
@typechecker
def wrong(a: F[Array, 'bs m']) -> DataWrong:
    _, m = a.shape
    return DataWrong(a[:, :m // 2], a[:, :m - 1])

def main():
    a = jnp.zeros((2, 4))
    correct(a)  # no exception

    try:
        wrong(a)  # should raise an exception
    except TypeError:
        pass
    else:
        raise RuntimeError('Expected an exception, but no exception is thrown')

if __name__ == '__main__':
    main()
