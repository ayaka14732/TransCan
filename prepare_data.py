import blingfire
from glob import glob
import json
from multiprocessing import Pool
import numpy as np
from transformers import BartTokenizer
from typing import List, Optional, Tuple

n_cpu: Optional[int] = 96
sequence_len: int = 15

def article_to_sentences(text: str) -> List[str]:
    '''
    ```python
    >>> article_to_sentences('A cat. The mouse.')
    ['A cat.', 'The mouse.']
    >>> article_to_sentences('A long line\nwith wrapping. The mouse.')
    ['A long line with wrapping.', 'The mouse.']
    >>> article_to_sentences('\n    ')
    []
    ```
    '''
    if not text.strip():
        return []
    return blingfire.text_to_sentences(text).split('\n')

def filename_to_sentences(filename: str) -> List[str]:
    all_sentences = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            text = obj['text']
            sentences = article_to_sentences(text)
            all_sentences.extend(sentences)
    return all_sentences

filenames = glob('./dump/*/*')[:10]

with Pool(processes=n_cpu) as p:
    list_sentences = p.map(filename_to_sentences, filenames)

def tokenize_sentences(sentences):
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    batch = tokenizer(sentences, max_length=sequence_len, padding='max_length', truncation=True, return_tensors='np')
    src = batch.input_ids
    mask_1d = batch.attention_mask.astype(np.bool_)
    return src, mask_1d

with Pool(processes=n_cpu) as p:
    list_results = p.map(tokenize_sentences, list_sentences)

def unzip(list_of_arrs: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    a = []
    b = []
    for x, y in list_of_arrs:
        a.append(x)
        b.append(y)
    a = np.vstack(a)
    b = np.vstack(b)
    return a, b

src, mask_1d = unzip(list_results)
