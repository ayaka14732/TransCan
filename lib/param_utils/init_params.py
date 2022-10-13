from transformers import BartConfig, FlaxBartModel

from .flax2jax import flax2jax
from ..random.wrapper import KeyArray, key2seed

def init_params(key: KeyArray) -> dict:
    seed = key2seed(key)
    config = BartConfig.from_pretrained('facebook/bart-base')
    model_flax = FlaxBartModel(config=config, seed=seed)
    model_jax = flax2jax(model_flax.params)
    return model_jax
