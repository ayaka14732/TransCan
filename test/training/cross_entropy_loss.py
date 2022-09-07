import jax; jax.config.update('jax_platforms', 'cpu'); jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)
from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import jax.numpy as np

from lib.training.cross_entropy_loss import cross_entropy_loss

# random key management boilerplate
seed = 42; from itertools import accumulate, chain, repeat; from operator import itemgetter; from lib.random.wrapper import seed2key, split_key, uniform; keys = map(itemgetter(0), accumulate(chain((split_key(seed2key(seed)),), repeat(None)), lambda acc, _: split_key(acc[1]))); rand = lambda *shape: uniform(next(keys), shape=shape)

batch_size = 3
seq_len = 2
n_classes = 10

logits = rand(batch_size, seq_len, n_classes)
labels = np.array([
    [4, 2],
    [1, 5],
    [8, 0],
], dtype=np.uint16)
assert np.all(labels >= 0) and np.all(labels < n_classes)
assert labels.shape == (batch_size, seq_len)
mask_dec_1d = np.ones((batch_size, seq_len), dtype=np.bool_)

loss = cross_entropy_loss(logits, labels, mask_dec_1d=mask_dec_1d, n_classes=n_classes)
