import jax
import jax.numpy as np
import jax.random as rand
from jaxtyping import Array as KeyArray

device_cpu = None

def do_on_cpu(f):
    global device_cpu
    if device_cpu is None:
        device_cpu = jax.devices('cpu')[0]

    def inner(*args, **kwargs):
        with jax.default_device(device_cpu):
            return f(*args, **kwargs)
    return inner

int32_min = np.iinfo(np.int32).min
int32_max = np.iinfo(np.int32).max
key2seed = do_on_cpu(lambda key: rand.randint(key, (), int32_min, int32_max).item())

seed2key = do_on_cpu(rand.PRNGKey)
seed2key.__doc__ = '''Same as `jax.random.PRNGKey`, but always produces the result on CPU.'''

split_key = do_on_cpu(rand.split)
split_key.__doc__ = '''Same as `jax.random.split`, but always produces the result on CPU.'''

# distributions

uniform = do_on_cpu(rand.uniform)
uniform.__doc__ = '''Same as `jax.random.uniform`, but always produces the result on CPU.'''

bernoulli = do_on_cpu(rand.bernoulli)
bernoulli.__doc__ = '''Same as `jax.random.bernoulli`, but always produces the result on CPU.'''
