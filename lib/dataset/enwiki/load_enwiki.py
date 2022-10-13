from glob import glob
from os.path import expanduser
from tqdm import tqdm

def load_enwiki(show_progress_bar=True) -> list[str]:
    filenames = glob(expanduser('~/.bart-base-jax/enwiki/dump2/*/*'))
    if not filenames:
        raise ValueError('Cannot find the dataset in ~/.bart-base-jax/enwiki/dump2.')

    sentences = []
    filenames_iter = filenames if not show_progress_bar else tqdm(filenames)
    for filename in filenames_iter:
        with open(filename, encoding='utf-8') as f:
            for line in f:
                sentences.append(line.rstrip('\n'))
    print(f'INFO: Loaded {len(sentences)} sentences.')
    return sentences
