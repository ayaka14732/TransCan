import os; os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_force_host_platform_device_count=2'
import jax; jax.config.update('jax_platforms', 'cpu')
from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import numpy as onp
from transformers import BartTokenizer

from lib.dataloader.prepare_data_for_model import prepare_data_for_model

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', add_prefix_space=True)

inputs_src = tokenizer(['<mask> go go', 'hi <mask>'], return_tensors='np', max_length=6, padding='max_length', truncation=True)
src = inputs_src.input_ids.astype(onp.uint16)
mask_enc_1d = inputs_src.attention_mask.astype(onp.bool_)

inputs_dst = tokenizer(['go go go', 'hi hi hi hi'], return_tensors='np', max_length=6, padding='max_length', truncation=True)
dst = inputs_dst.input_ids.astype(onp.uint16)
mask_dec_1d = inputs_dst.attention_mask.astype(onp.bool_)

batch_size, _ = src.shape
assert batch_size % jax.local_device_count() == 0

prepare_data_for_model(src, mask_enc_1d, dst, mask_dec_1d)  # type checking will happen
