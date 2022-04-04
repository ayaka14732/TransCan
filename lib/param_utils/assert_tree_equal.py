import jax.numpy as np
import numpy as onp

def assert_tree_equal(a, b, path=['root']) -> bool:
    if isinstance(a, int):
        assert isinstance(b, int), f'{type(b)}'
        assert a == b, f'{path}: {a} != {b}'

    elif isinstance(a, (np.ndarray, onp.ndarray)):
        assert isinstance(b, (np.ndarray, onp.ndarray)), f'{type(b)}'
        assert np.allclose(a, b), f'{path}: {a} != {b}'

    elif isinstance(a, dict):
        assert isinstance(b, dict), f'{path}: {type(a)} != {type(b)}'
        keys_a = sorted(a)
        keys_b = sorted(b)
        assert keys_a == keys_b
        for key in keys_a:
            assert_tree_equal(a[key], b[key], path=[*path, key])

    elif isinstance(a, list):
        assert isinstance(b, list), f'{path}: {type(a)} != {type(b)}'
        assert len(a) == len(b)
        for i, (a_, b_) in enumerate(zip(a, b)):
            assert_tree_equal(a_, b_, path=[*path, key])

    elif isinstance(a, tuple):
        assert isinstance(b, tuple), f'{path}: {type(a)} != {type(b)}'
        assert len(a) == len(b)
        for i, (a_, b_) in enumerate(zip(a, b)):
            assert_tree_equal(a_, b_, path=[*path, str(i)])

    else:
        raise NotImplementedError(f'Unsupported element type: {type(a)}')
