from jaxtyping import Array, Float as F, UInt16 as U16, PyTree, jaxtyped
from typeguard import check_type, typechecked as typechecker

@jaxtyped
@typechecker
def fwd_embedding(params: PyTree, x: U16[Array, '*dims']) -> F[Array, '*dims embed_size']:
    # params
    embedding: Array = params['embedding']  # array

    check_type('embedding', embedding, F[Array, 'vocab_size embed_size'])

    y = embedding[x]
    check_type('y', y, F[Array, '*dims embed_size'])

    return y
