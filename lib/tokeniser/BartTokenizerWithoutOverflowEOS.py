import numpy as onp
from transformers import BartTokenizer

class BartTokenizerWithoutOverflowEOS(BartTokenizer):
    def __call__(self, sentences, max_length):
        inputs = super().__call__(sentences, max_length=max_length-1, truncation=True, verbose=True, add_prefix_space=True, add_special_tokens=False)

        input_ids_ = []
        attention_masks_ = []

        for input_id, attention_mask in zip(inputs.input_ids, inputs.attention_mask):
            token_len = len(input_id)
            if token_len == max_length - 1:  # exceed `max_length - 1`, will not add `[EOS]`
                input_id = [self.bos_token_id, *input_id]
                attention_mask = [1, *attention_mask]
            else:  # will add `[EOS]`
                input_id = [self.bos_token_id, *input_id, self.eos_token_id, *(self.pad_token_id,) * (max_length - token_len - 2)]
                attention_mask = [1, *attention_mask, 1, *(0,) * (max_length - token_len - 2)]
            input_ids_.append(input_id)
            attention_masks_.append(attention_mask)

        input_ids = onp.array(input_ids_, dtype=onp.uint16)
        attention_masks = onp.array(attention_masks_, dtype=onp.bool_)

        return input_ids, attention_masks
