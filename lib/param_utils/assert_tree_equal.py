import jax.numpy as np
import numpy as onp

def assert_tree_equal(a, b) -> bool:
    if isinstance(a, (np.ndarray, onp.ndarray)):
        assert isinstance(b, (np.ndarray, onp.ndarray)), f'{type(b)}'
        assert np.allclose(a, b)

    elif isinstance(a, dict):
        assert isinstance(b, dict), f'{type(b)}'
        keys_a = sorted(a)
        keys_b = sorted(b)
        assert keys_a == keys_b
        for key in keys_a:
            assert_tree_equal(a[key], b[key])

    elif isinstance(a, list):
        assert isinstance(b, list), f'{type(b)}'
        assert len(a) == len(b)
        for a_, b_ in zip(a, b):
            assert_tree_equal(a_, b_)

    else:
        raise NotImplementedError(f'Unsupported element type: {type(a)}')
