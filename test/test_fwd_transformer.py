from itertools import accumulate, chain, repeat
import jax.numpy as np
from jax.random import PRNGKey, split, uniform
from operator import itemgetter

from fwd_transformer import fwd_transformer

seed = 42
keys = map(itemgetter(0), accumulate(chain((split(PRNGKey(seed)),), repeat(None)), lambda acc, _: split(acc[1])))
rand = lambda *shape: uniform(next(keys), shape)
