import os; os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_force_host_platform_device_count=8'
import jax; jax.config.update('jax_platforms', 'cpu')
from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as onp

from lib.preprocessor.device_split import device_split

if __name__ == '__main__':
    n_local_device = jax.local_device_count()
    assert n_local_device == 8

    a = onp.random.rand(n_local_device * 7, 20)
    a = device_split(a)
    assert a.shape == (n_local_device, 7, 20)
    assert [str(buffer.device()) for buffer in a.device_buffers] == [f'TFRT_CPU_{i}' for i in range(8)]
