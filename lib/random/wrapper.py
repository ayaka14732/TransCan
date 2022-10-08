import jax
import jax.numpy as np
import jax.random as rand
from jaxtyping import Array as KeyArray

int32_min = np.iinfo(np.int32).min
int32_max = np.iinfo(np.int32).max
key2seed = (lambda f: lambda x: f(x).item())(jax.jit(lambda key: rand.randint(key, (), int32_min, int32_max), backend='cpu'))

seed2key = jax.jit(rand.PRNGKey, backend='cpu')
seed2key.__doc__ = '''Same as `jax.random.PRNGKey`, but always produces the result on CPU.'''

split_key = jax.jit(rand.split, static_argnums=(1,), backend='cpu')
split_key.__doc__ = '''Same as `jax.random.split`, but always produces the result on CPU.'''

# distributions

uniform = jax.jit(rand.uniform, static_argnums=(1, 2), backend='cpu')
uniform.__doc__ = '''Same as `jax.random.uniform`, but always produces the result on CPU.'''

bernoulli = jax.jit(rand.bernoulli, static_argnums=(2,), backend='cpu')
bernoulli.__doc__ = '''Same as `jax.random.bernoulli`, but always produces the result on CPU.'''
