import jax
import jax.numpy as np
from jaxtyping import b as B, f as F, u32 as U32, jaxtyped
import optax
from typeguard import check_type, typechecked as typechecker

@jaxtyped
@typechecker
def cross_entropy_loss(
    logits: F['bs dst_len n_classes'],
    labels: U32['bs dst_len'],
    mask_dec_1d: B['bs dst_len'],
    n_classes: int,
) -> F['']:
    labels_onehot = jax.nn.one_hot(labels, num_classes=n_classes)
    check_type('labels_onehot', labels_onehot, F['bs dst_len n_classes'])

    loss = optax.softmax_cross_entropy(logits=logits, labels=labels_onehot)
    loss *= mask_dec_1d
    check_type('loss', loss, F['bs dst_len'])

    return np.sum(loss)
