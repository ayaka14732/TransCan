import jax.numpy as np

def fwd_layer_norm(params: dict, x: np.ndarray, eps: float=1e-5) -> np.ndarray:
    # params
    scale: np.ndarray = params['scale']  # array
    bias: np.ndarray = params['bias']  # array

    mean = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps)) * scale + bias
