import jax
import jax.numpy as np
from transformers import BartTokenizer

from lib.param_utils.load_params import load_params
from lib.generator import Generator

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

sentences = ['Can you see the beautiful flowers <mask> alongside the track?']
batch = tokenizer(sentences, return_tensors='jax')

src = batch.input_ids
mask_enc_1d = batch.attention_mask.astype(np.bool_)
mask_enc = np.einsum('bi,bj->bij', mask_enc_1d, mask_enc_1d)[:, None]

params = load_params('params_bart_base_en.dat')
params = jax.tree_map(np.asarray, params)

generator = Generator(params)
generate_ids = generator.generate(src, mask_enc)

decoded_sentences = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(decoded_sentences)
