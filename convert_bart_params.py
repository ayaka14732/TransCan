from transformers import FlaxBartForSequenceClassification

from lib.param_utils.save_params import save_params
from lib.param_utils.flax2jax import flax2jax

model = FlaxBartForSequenceClassification.from_pretrained('facebook/bart-base')
params = flax2jax(model.params['model'])
save_params(params, 'bart_params.dat')
