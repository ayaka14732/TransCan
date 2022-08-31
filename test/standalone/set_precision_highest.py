import jax
import jax.numpy as np
import numpy as onp

# https://github.com/google/jax/issues/9973#issuecomment-1073579382
jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)

x = onp.array([[0.3744, 0.1656],
               [0.4707, 0.1663]])
y = onp.array([[0.3946, 0.1186],
               [0.1569, 0.3145]])
z = onp.dot(x, y)

x_ = np.asarray(x)
y_ = np.asarray(y)
z_ = np.dot(x_, y_)

assert np.allclose(z, z_)
