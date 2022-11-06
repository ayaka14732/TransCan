import jax; jax.config.update('jax_platforms', 'cpu'); jax.config.update('jax_default_matmul_precision', jax.lax.Precision.HIGHEST)

import jax.numpy as np
import regex as re
import sys
from transformers import BartConfig, BartTokenizer, BertTokenizer
from tqdm import tqdm
from typing import Any

from lib.dataset.load_cantonese import load_cantonese
from lib.Generator import Generator
from lib.param_utils.load_params import load_params
from lib.en_kfw_nmt.fwd_transformer_encoder_part import fwd_transformer_encoder_part

def chunks(lst: list[Any], chunk_size: int) -> list[list[Any]]:
    '''Yield successive n-sized chunks from lst.'''
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def remove_tokenisation_space(s: str) -> str:
    '''
    >>> remove_space('阿 爸 好 忙 ， 成 日 出 差')
    '阿爸好忙，成日出差'
    >>> remove_space('摸 A B 12至 3')
    '摸A B 12至3'
    >>> remove_space('噉你哋要唔要呢 ？')
    '噉你哋要唔要呢？'
    >>> remove_space('3 . 1')
    '3.1'
    '''
    s = re.sub(r'(?<=[\p{Unified_Ideograph}\u3006\u3007。，、！：？（）《》「」]) (?=[\p{Unified_Ideograph}\u3006\u3007。，、！：？（）《》「」])', r'', s)
    s = re.sub(r'(?<=[\p{Unified_Ideograph}\u3006\u3007。，、！：？（）《》「」]) (?=[\da-zA-Z])', r'', s)
    s = re.sub(r'(?<=[\da-zA-Z]) (?=[\p{Unified_Ideograph}\u3006\u3007。，、！：？（）《》「」])', r'', s)
    s = re.sub(r'(?<=[\da-zA-Z]) (?=[.,])', r'', s)
    s = re.sub(r'(?<=[.,]) (?=[\da-zA-Z])', r'', s)
    return s

sentences = load_cantonese(split='test')
sentences_en = [en for en, _ in sentences]

param_file = sys.argv[1] if len(sys.argv) >= 2 else 'distinctive-gorge-3.dat'
params = load_params(param_file)
params = jax.tree_map(np.asarray, params)

tokenizer_en = BartTokenizer.from_pretrained('facebook/bart-base')
tokenizer_yue = BertTokenizer.from_pretrained('Ayaka/bart-base-cantonese')

config = BartConfig.from_pretrained('Ayaka/bart-base-cantonese')
generator = Generator({'embedding': params['decoder_embedding'], **params}, config=config)

predictions = []

for chunk in tqdm(chunks(sentences_en, chunk_size=32)):
    inputs = tokenizer_en(chunk, return_tensors='jax', padding=True)
    src = inputs.input_ids.astype(np.uint16)
    mask_enc_1d = inputs.attention_mask.astype(np.bool_)
    mask_enc = np.einsum('bi,bj->bij', mask_enc_1d, mask_enc_1d)[:, None]

    encoder_last_hidden_output = fwd_transformer_encoder_part(params, src, mask_enc)
    generated_ids = generator.generate(encoder_last_hidden_output, mask_enc_1d, num_beams=5, max_length=128)
    decoded_sentences = tokenizer_yue.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    for decoded_sentence in decoded_sentences:
        predictions.append(decoded_sentence)

with open('results-bart.txt', 'w', encoding='utf-8') as f:
    for prediction in predictions:
        prediction = remove_tokenisation_space(prediction)
        print(prediction, file=f)
