import jax.numpy as np
import numpy as onp
import tempfile
import torch
from transformers import BartConfig, BartForConditionalGeneration, FlaxBartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from .param_utils.jax2flax import jax2flax

'''
Notes: Tricky things

`FlaxBartForConditionalGeneration` has a `.encode` method to get the `encoder_last_hidden_output`,
but `BartForConditionalGeneration` does not support it.

`BartForConditionalGeneration`'s `.generate` method can accept a single `encoder_last_hidden_output`
parameter, but `FlaxBartForConditionalGeneration` does not support it.

So in the transformer library, both the PyTorch implementation and the Flax implementation
have some issues.
'''

class Generator:
    def __init__(self, params: np.ndarray, config: BartConfig=BartConfig.from_pretrained('facebook/bart-base')):
        # create a randomly initialized Flax model
        model_flax = FlaxBartForConditionalGeneration(config=config)

        # set the model parameter to the given parameter
        params_flax = jax2flax(params)
        model_flax.params['model'] = params_flax

        # save the Flax model and load it as a PyTorch model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_flax.save_pretrained(tmpdirname)
            model_pt = BartForConditionalGeneration.from_pretrained(tmpdirname, from_flax=True)

        self.params = params
        self.model = model_pt

    def generate(self, encoder_last_hidden_output: np.ndarray, **kwargs):
        encoder_outputs = BaseModelOutput(last_hidden_state=torch.from_numpy(onp.asarray(encoder_last_hidden_output)))
        generate_ids = self.model.generate(encoder_outputs=encoder_outputs, **kwargs)
        return np.asarray(generate_ids.numpy())
