from jaxtyping import Array, Bool as B, UInt16 as U16
from typing import NamedTuple

class Data(NamedTuple):
    src: U16[Array, 'n_local_devices bs//n_local_devices src_len']
    dst: U16[Array, 'n_local_devices bs//n_local_devices dst_len+1']
    mask_enc_1d: B[Array, 'n_local_devices bs//n_local_devices src_len']
    mask_dec_1d: B[Array, 'n_local_devices bs//n_local_devices dst_len+1']
    mask_enc: B[Array, 'n_local_devices bs//n_local_devices 1 src_len src_len']
    mask_dec: B[Array, 'n_local_devices bs//n_local_devices 1 dst_len+1 dst_len+1']
    mask_dec_enc: B[Array, 'n_local_devices bs//n_local_devices 1 dst_len+1 src_len']
    labels: U16[Array, 'n_local_devices bs//n_local_devices dst_len+1']
