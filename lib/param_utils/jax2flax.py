def convert_qkv(params):
    return {
        'kernel': params['kernel'].reshape(768, 768),
        'bias': params['bias'].reshape(768),
    }

def convert_transformer_encoder(params):
    return {
        'self_attn': {
            'q_proj': convert_qkv(params['self_attn']['q_proj']),
            'k_proj': convert_qkv(params['self_attn']['k_proj']),
            'v_proj': convert_qkv(params['self_attn']['v_proj']),
            'out_proj': params['self_attn']['ff'],
        },
        'self_attn_layer_norm': params['self_attn_layer_norm'],
        'fc1': params['ff0'],
        'fc2': params['ff1'],
        'final_layer_norm': params['final_layer_norm'],
    }

def convert_transformer_decoder(params):
    return {
        'self_attn': {
            'q_proj': convert_qkv(params['self_attn']['q_proj']),
            'k_proj': convert_qkv(params['self_attn']['k_proj']),
            'v_proj': convert_qkv(params['self_attn']['v_proj']),
            'out_proj': params['self_attn']['ff'],
        },
        'self_attn_layer_norm': params['self_attn_layer_norm'],
        'encoder_attn': {
            'q_proj': convert_qkv(params['cross_attn']['q_proj']),
            'k_proj': convert_qkv(params['cross_attn']['k_proj']),
            'v_proj': convert_qkv(params['cross_attn']['v_proj']),
            'out_proj': params['cross_attn']['ff'],
        },
        'encoder_attn_layer_norm': params['cross_attn_layer_norm'],
        'fc1': params['ff0'],
        'fc2': params['ff1'],
        'final_layer_norm': params['final_layer_norm'],
    }

def jax2flax(params):
    params = {
        'shared': {'embedding': params['embedding']['embedding']},
        'encoder': {
            'embed_positions': {'embedding': params['encoder_embed_positions']},
            'layernorm_embedding': params['encoder_embed_layer_norm'],
            'layers': {str(i): convert_transformer_encoder(layer) for i, layer in enumerate(params['encoder_layers'])},
        },
        'decoder': {
            'embed_positions': {'embedding': params['decoder_embed_positions']},
            'layernorm_embedding': params['decoder_embed_layer_norm'],
            'layers': {str(i): convert_transformer_decoder(layer) for i, layer in enumerate(params['decoder_layers'])},
        },
    }
    return params
