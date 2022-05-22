import jax
import tempfile
from transformers import BartForConditionalGeneration, FlaxBartForConditionalGeneration

from lib.param_utils.assert_tree_equal import assert_tree_equal
from lib.param_utils.save_params import save_params
from lib.param_utils.flax2jax import flax2jax
from lib.param_utils.pt2jax import pt2jax
from lib.param_utils.jax2flax import jax2flax

# facebook/bart-base

model_bart_base_en = FlaxBartForConditionalGeneration.from_pretrained('facebook/bart-base')
params_bart_base_en_flax = model_bart_base_en.params['model']
params_bart_base_en = flax2jax(params_bart_base_en_flax)
save_params(params_bart_base_en, 'params_bart_base_en.dat')

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

# randomly initialized fnlp/bart-base-chinese

model_bart_base_zh_rand = FlaxBartForConditionalGeneration(config=model_bart_base_zh.config, seed=42)
params_bart_base_zh_rand = flax2jax(model_bart_base_zh_rand.params['model'])

assert_tree_equal(
    jax.tree_map(lambda x: x.shape, params_bart_base_zh),
    jax.tree_map(lambda x: x.shape, params_bart_base_zh_rand),
)

save_params(params_bart_base_zh_rand, 'params_bart_base_zh_rand.dat')

# roundtrip test for flax2jax => jax2flax

params_bart_base_en_flax_roundtrip = jax2flax(params_bart_base_en)
assert_tree_equal(params_bart_base_en_flax, params_bart_base_en_flax_roundtrip)

# roundtrip test for flax2pt (done by the transformers library) => pt2jax and flax2jax

# save the Flax model and reload it as a PyTorch model
with tempfile.TemporaryDirectory() as tmpdirname:
    model_bart_base_zh_rand.save_pretrained(tmpdirname)
    model_bart_base_zh_rand_pt = BartForConditionalGeneration.from_pretrained(tmpdirname, from_flax=True)

params_bart_base_zh_rand_ = pt2jax(dict(model_bart_base_zh_rand_pt.model.named_parameters()))
assert_tree_equal(params_bart_base_zh_rand, params_bart_base_zh_rand_)
