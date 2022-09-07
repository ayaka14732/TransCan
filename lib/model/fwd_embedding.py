import jax.numpy as np
from jaxtyping import f as F, u16 as U16, PyTree, jaxtyped
from typeguard import check_type, typechecked as typechecker

@jaxtyped
@typechecker
def fwd_embedding(params: PyTree, x: U16['*dims']) -> F['*dims embed_size']:
    # params
    embedding: np.ndarray = params['embedding']  # array

    check_type('embedding', embedding, F['vocab_size embed_size'])

    y = embedding[x]
    check_type('y', y, F['*dims embed_size'])

    return y
