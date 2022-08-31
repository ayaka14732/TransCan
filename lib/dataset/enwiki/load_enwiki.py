from glob import glob
from os.path import expanduser
from tqdm import tqdm
from typing import List

def load_enwiki() -> List[str]:
    filenames = glob(expanduser('~/.bart-base-jax/enwiki/dump2/*/*'))
    if not filenames:
        raise ValueError('Cannot find the dataset in ~/.bart-base-jax/enwiki/dump2.')

    sentences = []
    for filename in tqdm(filenames):
        with open(filename, encoding='utf-8') as f:
            for line in f:
                sentences.append(line.rstrip('\n'))
    print(f'INFO: Loaded {len(sentences)} sentences.')
    return sentences
