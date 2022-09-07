import jax
import jax.numpy as np
from transformers import BartTokenizer
from typing import Tuple

from ..preprocessing.distort_sentence import distort_sentence
from ..random.wrapper import KeyArray, split_key

def tokenization_worker(x) -> np.ndarray:
    jax.config.update('jax_platforms', 'cpu')  # enforce CPU in subprocesses

    # sentences: list[str], key: KeyArray
    sentences, key = x

    global tokenizer
    if 'tokenizer' not in globals():
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    keys = split_key(key, num=len(sentences))
    distorted_sentences = [distort_sentence(sentence, key=key) for sentence, key in zip(sentences, keys)]

    x = tokenizer(sentences, return_tensors='np', max_length=256, padding='max_length', truncation=True, add_prefix_space=True)
    y = tokenizer(distorted_sentences, return_tensors='np', max_length=256, padding='max_length', truncation=True, add_prefix_space=True)

    src = x.input_ids.astype(np.uint32)
    mask_enc_1d = x.attention_mask.astype(np.bool_)
    dst = y.input_ids.astype(np.uint32)
    mask_dec_1d = y.attention_mask.astype(np.bool_)

    return src, mask_enc_1d, dst, mask_dec_1d
