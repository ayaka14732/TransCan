import jax; jax.config.update('jax_platforms', 'cpu'); jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)
from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import jax.numpy as np

from lib.model import fwd_embedding

# random key management boilerplate
seed = 42; from itertools import accumulate, chain, repeat; from operator import itemgetter; from lib.random.wrapper import seed2key, split_key, uniform; keys = map(itemgetter(0), accumulate(chain((split_key(seed2key(seed)),), repeat(None)), lambda acc, _: split_key(acc[1]))); rand = lambda *shape: uniform(next(keys), shape=shape)

vocab_size = 128
embed_size = 3

embedding = rand(vocab_size, embed_size)
x = np.array([0, 3, 15], dtype=np.uint16)

params = {'embedding': embedding}

output = fwd_embedding(params, x)

assert output.shape == (len(x), embed_size)
