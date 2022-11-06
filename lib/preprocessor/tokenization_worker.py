import jax.numpy as np

from ..tokeniser import EnTokenizer, YueTokenizer

tokenizer_en = tokenizer_yue = None

def tokenization_worker(sentences: list[tuple[str, str]]) -> np.ndarray:
    global tokenizer_en, tokenizer_yue
    if tokenizer_en is None:
        tokenizer_en = EnTokenizer.from_pretrained('facebook/bart-base')
        tokenizer_yue = YueTokenizer.from_pretrained('Ayaka/bart-base-cantonese')

    sentences_en = []
    sentences_yue = []
    for sentence_en, sentence_yue in sentences:
        sentences_en.append(sentence_en)
        sentences_yue.append(sentence_yue)

    max_length = 128
    src, mask_enc_1d = tokenizer_en(sentences_en, max_length=max_length)
    dst, mask_dec_1d = tokenizer_yue(sentences_yue, max_length=max_length-1)
    # TODO: add a reminder about these default settings:
    # - `return_tensors='np'`
    # - `add_prefix_space=True`
    # return type is a tuple, not a dict
    # return `np.uint16` and `np.bool_`

    return src, mask_enc_1d, dst, mask_dec_1d
