from glob import glob
import multiprocessing
from os import mkdir
from os.path import expanduser, exists, join
import subprocess
from tqdm import tqdm
from typing import List

from ...utils.download_file import download_file
from .preprocess_enwiki import preprocess_enwiki

def preprocess_enwiki_outer(f_in: str) -> None:
    dir_prefix, dump, dir_name, filename = f_in.rsplit('/', 3)
    assert dump == 'dump'
    dir_out = join(dir_prefix, 'dump2', dir_name)
    try:
        mkdir(dir_out)
    except FileExistsError:
        pass
    f_out = join(dir_out, filename)
    preprocess_enwiki(f_in, f_out)

def process_raw_files(raw_files: List[str]) -> None:
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool() as p:
        for _ in tqdm(p.imap_unordered(preprocess_enwiki_outer, raw_files), total=len(raw_files)):
            pass

def prepare_enwiki():
    # ~/.bart-base-jax
    root = expanduser('~/.bart-base-jax')
    if not exists(root):
        mkdir(root)

    # ~/.bart-base-jax/enwiki
    enwiki_path = join(root, 'enwiki')
    if not exists(enwiki_path):
        mkdir(enwiki_path)

    # ~/.bart-base-jax/enwiki/data.xml.bz2
    wikidata_name = 'data.xml.bz2'
    wikidata_path = join(enwiki_path, wikidata_name)
    if not exists(wikidata_path):
        download_file('https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2', wikidata_name, dir_prefix=enwiki_path)

    # ~/.bart-base-jax/enwiki/dump
    enwiki_dump_path = join(enwiki_path, 'dump')
    if not exists(enwiki_dump_path):
        subprocess.call(['python', '-m', 'wikiextractor.WikiExtractor', wikidata_path, '--json', '-o', enwiki_dump_path])

    # ~/.bart-base-jax/enwiki/dump2
    enwiki_dump2_path = join(enwiki_path, 'dump2')
    if not exists(enwiki_dump2_path):
        mkdir(enwiki_dump2_path)
        raw_files = glob(join(enwiki_dump_path, '*/*'))
        assert raw_files, f'Expected files in {enwiki_dump_path}'
        process_raw_files(raw_files)
