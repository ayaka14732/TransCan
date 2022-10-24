import jax; jax.config.update('jax_platforms', 'cpu'); jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)
from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from lib.model import fwd_linear

# random key management boilerplate
seed = 42; from itertools import accumulate, chain, repeat; from operator import itemgetter; from lib.random.wrapper import seed2key, split_key, uniform; keys = map(itemgetter(0), accumulate(chain((split_key(seed2key(seed)),), repeat(None)), lambda acc, _: split_key(acc[1]))); rand = lambda *shape: uniform(next(keys), shape=shape)

x = rand(3, 2, 5)
kernel = rand(5, 4)
bias = rand(4)

output = fwd_linear({'kernel': kernel, 'bias': bias}, x)
