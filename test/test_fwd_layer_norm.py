import flax.linen as fnn
from itertools import accumulate, chain, repeat
import jax
import jax.numpy as np
from jax.random import PRNGKey, split, uniform
from operator import itemgetter

from fwd_layer_norm import fwd_layer_norm

# https://github.com/google/jax/issues/9973#issuecomment-1073579382
jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)

seed = 42
keys = map(itemgetter(0), accumulate(chain((split(PRNGKey(seed)),), repeat(None)), lambda acc, _: split(acc[1])))
rand = lambda *shape: uniform(next(keys), shape)

x = rand(3, 2, 5)
scale = rand(5)
bias = rand(5)

params = {'scale': scale, 'bias': bias}

output = fwd_layer_norm(params, x)

# Flax implementation

model = fnn.LayerNorm(epsilon=1e-5)

output_ = model.apply({'params': {'scale': scale, 'bias': bias}}, x)

assert np.allclose(output, output_)
