import jax.random as rand
from jaxtyping import Array, Shaped as S, jaxtyped

from ..random.wrapper import KeyArray

@jaxtyped
def dropout(key: KeyArray, x: S[Array, '*dims'], keep_rate: float=0.9) -> S[Array, '*dims']:
    assert 0. <= keep_rate <= 1.
    y = x * rand.bernoulli(key, p=keep_rate, shape=x.shape) / keep_rate
    return y
