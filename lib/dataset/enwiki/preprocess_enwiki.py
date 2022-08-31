import blingfire
import json
from typing import List

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

def preprocess_enwiki(filename_in: str, filename_out: str) -> List[str]:
    all_sentence = []
    with open(filename_in, encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            text = obj['text']
            sentences = article_to_sentences(text)
            for sentence in sentences:
                if sentence.count(' ') < 2:  # zero or one word
                    continue
                all_sentence.append(sentence)
    with open(filename_out, 'w', encoding='utf-8') as f:
        for sentence in all_sentence:
            assert '\n' not in sentence
            print(sentence, file=f)
