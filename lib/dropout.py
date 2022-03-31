import jax.numpy as np
import jax.random as rand

def dropout(key: rand.KeyArray, x: np.ndarray):
    keep_rate = 0.9

    y = x * rand.bernoulli(key, p=keep_rate, shape=x.shape) / keep_rate
    return y
