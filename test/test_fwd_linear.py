from itertools import accumulate, chain, repeat
import jax
from jax.random import PRNGKey, split, uniform
from operator import itemgetter

from fwd_linear import fwd_linear

# https://github.com/google/jax/issues/9973#issuecomment-1073579382
jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)

seed = 42
keys = map(itemgetter(0), accumulate(chain((split(PRNGKey(seed)),), repeat(None)), lambda acc, _: split(acc[1])))
rand = lambda *shape: uniform(next(keys), shape)

x = rand(3, 2, 5)
kernel = rand(5, 4)
bias = rand(4)

output = fwd_linear({'kernel': kernel, 'bias': bias}, x)
