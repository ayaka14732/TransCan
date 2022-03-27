from itertools import accumulate, chain, repeat
import jax.numpy as np
from jax.random import PRNGKey, split, uniform
from operator import itemgetter

from fwd_embedding import fwd_embedding

seed = 42
keys = map(itemgetter(0), accumulate(chain((split(PRNGKey(seed)),), repeat(None)), lambda acc, _: split(acc[1])))
rand = lambda *shape: uniform(next(keys), shape)

vocab_size = 128
embed_size = 3

embedding = rand(vocab_size, embed_size)
x = np.array([0, 3, 15])

params = {'embedding': embedding}

output = fwd_embedding(params, x)

assert output.shape == (len(x), embed_size)
