import jax
from transformers import BartForConditionalGeneration, FlaxBartForSequenceClassification

from lib.param_utils.assert_tree_equal import assert_tree_equal
from lib.param_utils.save_params import save_params
from lib.param_utils.flax2jax import flax2jax
from lib.param_utils.pt2jax import pt2jax

# facebook/bart-base

model_bart_base_en = FlaxBartForSequenceClassification.from_pretrained('facebook/bart-base')
params_bart_base_en = flax2jax(model_bart_base_en.params['model'])
# save_params(params_bart_base_en, 'params_bart_base_en.dat')

# fnlp/bart-base-chinese

model_bart_base_zh = BartForConditionalGeneration.from_pretrained('fnlp/bart-base-chinese')
params_bart_base_zh = pt2jax(dict(model_bart_base_zh.model.named_parameters()))

def check_shape_equal_except_for_embeddings():
    def inner(x):
        embed_sizes = (
            # vocab size (50265 vs 21128)
            model_bart_base_en.config.vocab_size,
            model_bart_base_zh.config.vocab_size,
            # max position embeddings (1026 vs 514)
            model_bart_base_en.config.max_position_embeddings + 2,
            model_bart_base_zh.config.max_position_embeddings + 2,
        )
        return tuple(-1 if i in embed_sizes else i for i in x.shape)
    assert_tree_equal(
        jax.tree_map(inner, params_bart_base_en),
        jax.tree_map(inner, params_bart_base_zh),
    )
check_shape_equal_except_for_embeddings()

save_params(params_bart_base_zh, 'params_bart_base_zh.dat')

