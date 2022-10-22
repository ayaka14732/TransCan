from jaxtyping import Bool as B, UInt16 as U16, jaxtyped
import numpy as onp
from typeguard import typechecked

from .device_split import device_split
from .Data import Data

@jaxtyped
@typechecked
def prepare_data_for_model(
    src: U16[onp.ndarray, 'bs src_len'],
    mask_enc_1d: B[onp.ndarray, 'bs src_len'], 
    dst: U16[onp.ndarray, 'bs dst_len'],
    mask_dec_1d: B[onp.ndarray, 'bs dst_len'],
) -> Data:
    # TODO: is this part correct?
    batch_size, _ = dst.shape

    labels = dst

    prepend_eos_for_dst = True
    bos_id = 2

    if prepend_eos_for_dst:
        arr_eos = onp.ones((batch_size, 1), dtype=onp.uint16) * bos_id
        dst = onp.hstack((arr_eos, dst))

        arr_true = onp.ones((batch_size, 1), dtype=onp.bool_)
        mask_dec_1d = onp.hstack((arr_true, mask_dec_1d))

        arr_whatever = onp.ones((batch_size, 1), dtype=onp.uint16)
        labels = onp.hstack((labels, arr_whatever))
    # end todo

    mask_enc = onp.einsum('bi,bj->bij', mask_enc_1d, mask_enc_1d)[:, None]
    mask_dec = onp.tril(onp.einsum('bi,bj->bij', mask_dec_1d, mask_dec_1d))[:, None]
    mask_dec_enc = onp.einsum('bi,bj->bij', mask_dec_1d, mask_enc_1d)[:, None]

    data = src, dst, mask_enc_1d, mask_dec_1d, mask_enc, mask_dec, mask_dec_enc, labels
    return Data(*map(device_split, data))
