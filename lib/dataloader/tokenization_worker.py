from typing import Callable
import jax.numpy as np
from transformers import BartTokenizer
from wakong import Wakong

from ..random.wrapper import KeyArray, key2seed

tokenizer = None

def distort_sentence(wakong: Callable, sentence: str) -> str:
    words = sentence.split(' ')  # TODO: possibility to use Blingfire?
    masked_words = wakong(words, mask_token='<mask>')
    return ' '.join(masked_words)

def tokenization_worker(x) -> np.ndarray:
    # sentences: list[str], key: KeyArray
    sentences, key = x

    seed = key2seed(key)  # TODO: optimisation: avoid seed conversion on every function call
    wakong = Wakong(seed=seed)

    global tokenizer
    if tokenizer is None:
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    distorted_sentences = [distort_sentence(wakong, sentence) for sentence in sentences]

    x = tokenizer(distorted_sentences, return_tensors='np', max_length=256, padding='max_length', truncation=True, add_prefix_space=True)
    y = tokenizer(sentences, return_tensors='np', max_length=256-1, padding='max_length', truncation=True, add_prefix_space=True)

    src = x.input_ids.astype(np.uint16)
    mask_enc_1d = x.attention_mask.astype(np.bool_)
    dst = y.input_ids.astype(np.uint16)
    mask_dec_1d = y.attention_mask.astype(np.bool_)

    return src, mask_enc_1d, dst, mask_dec_1d
