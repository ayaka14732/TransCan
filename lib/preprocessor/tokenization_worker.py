from typing import Callable
import jax.numpy as np
from wakong import Wakong

from ..random.wrapper import KeyArray, key2seed
from ..tokeniser import BartTokenizerWithoutOverflowEOS

tokenizer = None

def distort_sentence(wakong: Callable, sentence: str) -> str:
    words = sentence.split(' ')  # TODO: possibility to use Blingfire?
    masked_words = wakong(words, mask_token='<mask>')
    return ' '.join(masked_words)

def tokenization_worker(sentences: list[str], key: KeyArray) -> np.ndarray:
    seed = key2seed(key)  # TODO: optimisation: avoid seed conversion on every function call
    wakong = Wakong(seed=seed)

    global tokenizer
    if tokenizer is None:
        tokenizer = BartTokenizerWithoutOverflowEOS.from_pretrained('facebook/bart-base')

    distorted_sentences = [distort_sentence(wakong, sentence) for sentence in sentences]

    src, mask_enc_1d = tokenizer(distorted_sentences, max_length=256)
    dst, mask_dec_1d = tokenizer(sentences, max_length=256-1)
    # TODO: add a reminder about these default settings:
    # - `return_tensors='np'`
    # - `add_prefix_space=True`
    # return type is a tuple, not a dict
    # return `np.uint16` and `np.bool_`

    return src, mask_enc_1d, dst, mask_dec_1d
