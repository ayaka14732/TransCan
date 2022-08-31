import jax; jax.config.update('jax_platforms', 'cpu'); jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)
from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import flax.linen as fnn
import jax.numpy as np

from lib.model.fwd_layer_norm import fwd_layer_norm

# random key management boilerplate
seed = 42; from itertools import accumulate, chain, repeat; from operator import itemgetter; from lib.random.wrapper import seed2key, split_key, uniform; keys = map(itemgetter(0), accumulate(chain((split_key(seed2key(seed)),), repeat(None)), lambda acc, _: split_key(acc[1]))); rand = lambda *shape: uniform(next(keys), shape=shape)

x = rand(3, 2, 5)
scale = rand(5)
bias = rand(5)

params = {'scale': scale, 'bias': bias}

output = fwd_layer_norm(params, x, eps=1e-5)

# Flax implementation

model = fnn.LayerNorm(epsilon=1e-5)

output_ = model.apply({'params': {'scale': scale, 'bias': bias}}, x)

assert np.allclose(output, output_)
