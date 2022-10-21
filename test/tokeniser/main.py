import jax; jax.config.update('jax_platforms', 'cpu')
from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from lib.tokeniser import BartTokenizerWithoutOverflowEOS

tokenizer = BartTokenizerWithoutOverflowEOS.from_pretrained('facebook/bart-base')

sentences = ['a a', 'go go go', 'hi hi hi hi', 'ox ox ox ox ox']
max_length = 5
input_ids, attention_masks = tokenizer(sentences, max_length)

assert input_ids.tolist() == \
    [[0, 10, 10, 2, 1],
    [0, 213, 213, 213, 2],
    [0, 20280, 20280, 20280, 20280],
    [0, 33665, 33665, 33665, 33665]]
assert attention_masks.tolist() == \
    [[True, True, True, True, False],
    [True, True, True, True, True],
    [True, True, True, True, True],
    [True, True, True, True, True]]
