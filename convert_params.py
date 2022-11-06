import jax; jax.config.update('jax_platforms', 'cpu')
import flax.linen as nn
import jax.numpy as np
from transformers import FlaxBartModel

from lib.param_utils.flax2jax import flax2jax
from lib.param_utils.save_params import save_params
from lib.random.wrapper import seed2key, split_key

model_en = FlaxBartModel.from_pretrained('facebook/bart-base')
params_en = flax2jax(model_en.params)

model_yue = FlaxBartModel.from_pretrained('Ayaka/bart-base-cantonese')
params_yue = flax2jax(model_yue.params)

# linear projection layers are inserted between
# encoder layer -2 and -1 (-1 is the last layer)
seed = 10138  # random seed
key = seed2key(seed)

# 768 -> 1024
key, subkey = split_key(key)
proj = nn.Dense(1024)
params_proj0 = proj.init(subkey, np.ones((1, 1, 768)))['params']
params_proj0 = {'kernel': params_proj0['kernel'], 'bias': params_proj0['bias']}

# 1024 -> 768
key, subkey = split_key(key)
proj = nn.Dense(768)
params_proj1 = proj.init(subkey, np.ones((1, 1, 1024)))['params']
params_proj1 = {'kernel': params_proj1['kernel'], 'bias': params_proj1['bias']}

params = {
    'encoder_embedding': params_en['embedding'],
    'encoder_embed_positions': params_en['encoder_embed_positions'],
    'encoder_embed_layer_norm': params_en['encoder_embed_layer_norm'],
    'encoder_layers': params_en['encoder_layers'][:-1] + params_yue['encoder_layers'][-1:],
    'proj0': params_proj0,
    'proj1': params_proj1,
    'decoder_embedding': params_yue['embedding'],
    'decoder_embed_positions': params_yue['decoder_embed_positions'],
    'decoder_embed_layer_norm': params_yue['decoder_embed_layer_norm'],
    'decoder_layers': params_yue['decoder_layers'],
    'lm_head': params_yue['embedding']['embedding'].T,
}

save_params(params, 'params_merged.dat')
