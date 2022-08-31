import jax; jax.config.update('jax_platforms', 'cpu'); jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)

import jax.numpy as np

a = np.ones((4, 8))
assert 'CPU' in str(a.device())
