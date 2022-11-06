import jax.numpy as np
import numpy as onp
from transformers import BertTokenizer
from typing import Literal

class YueTokenizer(BertTokenizer):
    def __call__(self, sentences, max_length, return_tensors: Literal['np', 'jax']='np'):
        if return_tensors not in ('np', 'jax'):
            raise ValueError(f"`return_tensors` should be one of ('np', 'jax')")

        inputs = super().__call__(sentences, max_length=max_length-1, truncation=True, verbose=True, add_special_tokens=False)

        input_ids_ = []
        attention_masks_ = []

        for input_id, attention_mask in zip(inputs.input_ids, inputs.attention_mask):
            token_len = len(input_id)
            if token_len == max_length - 1:  # exceed `max_length - 1`, will not add `[SEP]`
                input_id = [self.cls_token_id, *input_id]
                attention_mask = [1, *attention_mask]
            else:  # will add `[SEP]`
                input_id = [self.cls_token_id, *input_id, self.sep_token_id, *(self.pad_token_id,) * (max_length - token_len - 2)]
                attention_mask = [1, *attention_mask, 1, *(0,) * (max_length - token_len - 2)]
            input_ids_.append(input_id)
            attention_masks_.append(attention_mask)

        array = np.array if return_tensors == 'jax' else onp.array

        input_ids = array(input_ids_, dtype=onp.uint16)
        attention_masks = array(attention_masks_, dtype=onp.bool_)

        return input_ids, attention_masks
