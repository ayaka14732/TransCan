from os.path import abspath, dirname, join
from typing import List

here = dirname(abspath(__file__))

def load_dummy() -> List[str]:
    english_sentences = {}

    with open(join(here, 'tatoeba-uyghur-english-2022-08-28.tsv')) as f:
        for line in f:
            _, _, _, english = line.rstrip('\n').split('\t')
            english_sentences[english] = None

    return list(english_sentences)
