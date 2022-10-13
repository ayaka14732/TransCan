import jax
from jaxtyping import Array, Bool as B, Shaped as S, UInt16 as U16, jaxtyped
import multiprocessing
import numpy as onp
import random
from typeguard import typechecked as typechecker
from typing import Any, NamedTuple, Optional

from .ProcessPoolExecutorWithQueueSizeLimit import ProcessPoolExecutorWithQueueSizeLimit
from .tokenization_worker import tokenization_worker
from ..dataset.dummy.load_dummy import load_dummy
from ..dataset.enwiki.load_enwiki import load_enwiki
from ..random.wrapper import KeyArray, key2seed, split_key

class Data(NamedTuple):
    src: Array
    dst: Array
    mask_enc_1d: Array
    mask_dec_1d: Array
    mask_enc: Array
    mask_dec: Array
    mask_dec_enc: Array
    labels: Array

@jaxtyped
@typechecker
def device_split(a: S[onp.ndarray, '...']) -> S[Array, '...']:
    '''Splits the first axis of `a` evenly across the number of devices.'''
    local_devices = jax.local_devices()
    n_local_devices = jax.local_device_count()

    batch_size, *shapes = a.shape
    a = a.reshape(n_local_devices, batch_size // n_local_devices, *shapes)
    b = jax.device_put_sharded(tuple(a), devices=local_devices)
    return b

@jaxtyped
@typechecker
def make_data(
    src: U16[onp.ndarray, 'bs src_len'],
    mask_enc_1d: B[onp.ndarray, 'bs src_len'], 
    dst: U16[onp.ndarray, 'bs dst_len'],
    mask_dec_1d: B[onp.ndarray, 'bs dst_len'],
) -> Data:
    # TODO: is this part correct?
    labels = dst

    batch_size, *_ = dst.shape

    bos_id = 2

    eoss = onp.ones((batch_size, 1), dtype=onp.uint16) * bos_id
    dst = onp.hstack((eoss, dst[:, 1:]))

    trues = onp.ones((batch_size, 1), dtype=onp.bool_)
    mask_dec_1d = onp.hstack((trues, mask_dec_1d[:, 1:]))
    # end todo

    mask_enc = onp.einsum('bi,bj->bij', mask_enc_1d, mask_enc_1d)[:, None]
    mask_dec = onp.tril(onp.einsum('bi,bj->bij', mask_dec_1d, mask_dec_1d))[:, None]
    mask_dec_enc = onp.einsum('bi,bj->bij', mask_dec_1d, mask_enc_1d)[:, None]

    d = src, dst, mask_enc_1d, mask_dec_1d, mask_enc, mask_dec, mask_dec_enc, labels
    return Data(*map(device_split, d))

def chunks(lst: list[Any], chunk_size: int) -> list[list[Any]]:
    '''Yield successive n-sized chunks from lst.'''
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

class DataLoader:
    def __init__(self, dataset: str, key: KeyArray, batch_size_per_device: int, n_workers: Optional[int]=None, queue_size: int=64, chunk_size: Optional[int]=1024, should_shuffle: bool=True):
        process_index = jax.process_index()
        n_local_devices = jax.local_device_count()

        if dataset == 'enwiki':
            sentences = load_enwiki(show_progress_bar=process_index == 0)
        elif dataset == 'dummy':
            sentences = load_dummy()
        else:
            raise ValueError(f'Invalid dataset: {repr(dataset)}')

        batch_size = batch_size_per_device * n_local_devices

        self.sentences = sentences
        self.key = key
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.queue_size = queue_size
        self.chunk_size = chunk_size
        self.should_shuffle = should_shuffle

    def __iter__(self):
        process_index = jax.process_index()
        process_count = jax.process_count()

        if self.should_shuffle:
            self.key, subkey = split_key(self.key)
            seed = key2seed(subkey)
            rng = random.Random(seed)
            rng.shuffle(sentences)

        # TODO: is it plausible to split sentences at preprocessing time?
        sentences_per_device = len(sentences) // process_count
        sentences = sentences[process_index * sentences_per_device:(process_index + 1) * sentences_per_device]

        sentences_chunked = chunks(sentences, chunk_size=self.chunk_size)
        n_sentences = len(sentences)
        n_chunks = len(sentences_chunked)
        print(f'INFO: Successfully split {n_sentences} sentences into {n_chunks} chunks.')

        ctx = multiprocessing.get_context('spawn')
        with ProcessPoolExecutorWithQueueSizeLimit(queue_size=self.queue_size, max_workers=self.n_workers, mp_context=ctx) as executor:
            self.key, *subkeys = split_key(self.key, num=n_chunks)
            results = executor.map(tokenization_worker, zip(sentences_chunked, subkeys))

            src_ = None
            mask_enc_1d_ = None
            dst_ = None
            mask_dec_1d_ = None

            for src, mask_enc_1d, dst, mask_dec_1d in results:
                if src_ is not None:
                    src = onp.vstack((src_, src))
                    mask_enc_1d = onp.vstack((mask_enc_1d_, mask_enc_1d))
                    dst = onp.vstack((dst_, dst))
                    mask_dec_1d = onp.vstack((mask_dec_1d_, mask_dec_1d))

                while True:
                    if src.shape[0] < self.batch_size:
                        src_ = src
                        mask_enc_1d_ = mask_enc_1d
                        dst_ = dst
                        mask_dec_1d_ = mask_dec_1d

                        break

                    elif src.shape[0] == self.batch_size:
                        src_ = None
                        mask_enc_1d_ = None
                        dst_ = None
                        mask_dec_1d_ = None

                        yield make_data(src, mask_enc_1d, dst, mask_dec_1d)
                        break

                    else:
                        src_ = None
                        mask_enc_1d_ = None
                        dst_ = None
                        mask_dec_1d_ = None

                        yield make_data(src[:self.batch_size], mask_enc_1d[:self.batch_size], dst[:self.batch_size], mask_dec_1d[:self.batch_size])

                        src = src[self.batch_size:]
                        mask_enc_1d = mask_enc_1d[self.batch_size:]
                        dst = dst[self.batch_size:]
                        mask_dec_1d = mask_dec_1d[self.batch_size:]
