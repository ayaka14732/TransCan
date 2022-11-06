import jax; jax.config.update('jax_platforms', 'cpu')

import jax.numpy as np

a = np.array([1., 2., 3.])
mask = np.array([1, 1, 0], dtype=np.bool_)
assert a.mean(where=mask).item() == 1.5
