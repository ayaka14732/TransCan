import jax
from jaxtyping import Array, jaxtyped
import numpy as onp
from typeguard import typechecked as typechecker

devices = jax.devices()
n_devices = jax.device_count()

@jaxtyped
@typechecker
def device_split(a: onp.ndarray) -> Array['...']:
    '''Splits the first axis of `a` evenly across the number of devices.'''
    batch_size, *shapes = a.shape
    a = a.reshape(n_devices, batch_size // n_devices, *shapes)
    b = jax.device_put_sharded(tuple(a), devices=devices)
    return b
