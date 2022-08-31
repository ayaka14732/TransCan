import jax.numpy as np
import numpy as onp
import tempfile
import torch
from transformers import BartConfig, BartForConditionalGeneration, FlaxBartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from .param_utils.flax2jax import flax2jax
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
        # params
        embedding: dict = params['embedding']  # embedding
        decoder_embed_positions: np.ndarray = params['decoder_embed_positions']  # array
        decoder_embed_layer_norm: dict = params['decoder_embed_layer_norm']  # layer norm
        decoder_layers: list = params['decoder_layers']  # list of transformer encoder

        # randomly initialize a Flax model
        model_flax = FlaxBartForConditionalGeneration(config=config)

        # extract model parameters and convert to JAX
        params_jax = flax2jax(model_flax.params['model'])

        # set the decoder part of the model parameters to the given parameters
        params_jax = {
            **params_jax,
            'embedding': embedding,
            'decoder_embed_positions': decoder_embed_positions,
            'decoder_embed_layer_norm': decoder_embed_layer_norm,
            'decoder_layers': decoder_layers,
        }

        # convert back to Flax
        params_flax = jax2flax(params_jax)
        model_flax.params['model'] = params_flax

        # save the Flax model and reload it as a PyTorch model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_flax.save_pretrained(tmpdirname)
            model_pt = BartForConditionalGeneration.from_pretrained(tmpdirname, from_flax=True)

        self.model = model_pt

    def generate(self, encoder_last_hidden_output: np.ndarray, mask_enc_1d: np.ndarray, **kwargs):
        encoder_outputs = BaseModelOutput(last_hidden_state=torch.from_numpy(onp.asarray(encoder_last_hidden_output)))
        attention_mask = torch.from_numpy(onp.asarray(mask_enc_1d))
        generate_ids = self.model.generate(attention_mask=attention_mask, encoder_outputs=encoder_outputs, **kwargs)
        return np.asarray(generate_ids.numpy())
