from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import jax
import jax.numpy as np
import sys

from lib.random.wrapper import do_on_cpu, seed2key, uniform

def get_device_name(a: np.ndarray) -> str:
    return str(a.device())

device = jax.devices()[0]
if 'CPU' in str(device):
    print('Warning: no GPU/TPU device is found, skipping tests...')
    sys.exit(0)

def f():
    a = np.array([1., 2.])
    return a

assert 'CPU' not in get_device_name(f())
assert 'CPU' in get_device_name(do_on_cpu(f)())

key = seed2key(42)

assert 'CPU' in get_device_name(key)
assert 'CPU' in get_device_name(uniform(key))
