import jax; jax.config.update('jax_platforms', 'cpu'); jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)
from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import contextlib
import io
import jax.numpy as np

from lib.debug.log import log, log_shape

arr = np.zeros((10, 10))

f = io.StringIO()
with contextlib.redirect_stdout(f):
    log('bbbcUBxRLnhSxSXLNEcDAzgPMDKFnJCgoQKGUKcYdpcQLPfj')  # test `log`
    log_shape('arr', arr)  # test `log_shape`

assert f.getvalue() == '''[log.py:14] bbbcUBxRLnhSxSXLNEcDAzgPMDKFnJCgoQKGUKcYdpcQLPfj
[log.py:15] arr: (10, 10)
'''
