import jax
from jaxtyping import Array, Shaped as S, jaxtyped
import numpy as onp
from typeguard import typechecked

@jaxtyped
@typechecked
def device_split(a: S[onp.ndarray, '...']) -> S[Array, '...']:
    '''Splits the first axis of `a` evenly across all local devices.'''
    local_devices = jax.local_devices()
    n_local_devices = jax.local_device_count()

    batch_size, *shapes = a.shape
    a = a.reshape(n_local_devices, batch_size // n_local_devices, *shapes)
    b = jax.device_put_sharded(tuple(a), devices=local_devices)
    return b
