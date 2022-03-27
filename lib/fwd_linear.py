import jax.numpy as np

def fwd_linear(params: dict, x: np.ndarray) -> np.ndarray:
    # params
    kernel: np.ndarray = params['kernel']  # array
    bias: np.ndarray = params['bias']  # array

    return np.dot(x, kernel) + bias
