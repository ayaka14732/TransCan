import jax.numpy as np

def fwd_embedding(params: dict, x: np.ndarray) -> np.ndarray:
    # params
    embedding: np.ndarray = params['embedding']  # array

    y = embedding[x]
    return y
