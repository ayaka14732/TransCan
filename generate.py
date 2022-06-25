import jax
import jax.numpy as np
from transformers import BartTokenizer

from lib.param_utils.load_params import load_params
from lib.fwd_embedding import fwd_embedding
from lib.fwd_layer_norm import fwd_layer_norm
from lib.fwd_transformer_encoder import fwd_transformer_encoder
from lib.generator import Generator

def fwd_encode(params: dict, src: np.ndarray, mask_enc: np.ndarray) -> np.ndarray:
    # params
    embedding: dict = params['embedding']  # embedding
    encoder_embed_positions: np.ndarray = params['encoder_embed_positions']  # array
    encoder_embed_layer_norm: dict = params['encoder_embed_layer_norm']  # layer norm
    encoder_layers: list = params['encoder_layers']  # list of transformer encoder

    _, width_enc = src.shape

    offset = 2

    # encoder
    src = fwd_embedding(embedding, src)
    src = src + encoder_embed_positions[offset:width_enc+offset]
    src = fwd_layer_norm(encoder_embed_layer_norm, src)

    for encoder_layer in encoder_layers:
        src = fwd_transformer_encoder(encoder_layer, src, mask_enc)

    return src

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

sentences = ['Can you see the beautiful flowers <mask> alongside the track?', 'Upon graduation, <mask> herself.']
batch = tokenizer(sentences, padding=True, return_tensors='jax')

src = batch.input_ids
mask_enc_1d = batch.attention_mask.astype(np.bool_)
mask_enc = np.einsum('bi,bj->bij', mask_enc_1d, mask_enc_1d)[:, None]

params = load_params('params_bart_base_en.dat')
params = jax.tree_map(np.asarray, params)

encoder_last_hidden_output = fwd_encode(params, src, mask_enc)

generator = Generator(params)
generate_ids = generator.generate(encoder_last_hidden_output, mask_enc_1d, num_beams=5)

decoded_sentences = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(decoded_sentences)
