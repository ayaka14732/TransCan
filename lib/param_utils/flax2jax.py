def convert_qkv(params):
    return {
        'kernel': params['kernel'].reshape(768, 12, 64),
        'bias': params['bias'].reshape(12, 64),
    }

def convert_ff(params):
    return {
        'kernel': params['kernel'].reshape(12, 64, 768),
        'bias': params['bias'],
    }

def convert_transformer_encoder(params):
    return {
        'self_attn': {
            'q_proj': convert_qkv(params['self_attn']['q_proj']),
            'k_proj': convert_qkv(params['self_attn']['k_proj']),
            'v_proj': convert_qkv(params['self_attn']['v_proj']),
            'ff': convert_ff(params['self_attn']['out_proj']),
        },
        'self_attn_layer_norm': params['self_attn_layer_norm'],
        'ff0': params['fc1'],
        'ff1': params['fc2'],
        'final_layer_norm': params['final_layer_norm'],
    }

def convert_transformer_decoder(params):
    return {
        'self_attn': {
            'q_proj': convert_qkv(params['self_attn']['q_proj']),
            'k_proj': convert_qkv(params['self_attn']['k_proj']),
            'v_proj': convert_qkv(params['self_attn']['v_proj']),
            'ff': convert_ff(params['self_attn']['out_proj']),
        },
        'self_attn_layer_norm': params['self_attn_layer_norm'],
        'cross_attn': {
            'q_proj': convert_qkv(params['encoder_attn']['q_proj']),
            'k_proj': convert_qkv(params['encoder_attn']['k_proj']),
            'v_proj': convert_qkv(params['encoder_attn']['v_proj']),
            'ff': convert_ff(params['encoder_attn']['out_proj']),
        },
        'cross_attn_layer_norm': params['encoder_attn_layer_norm'],
        'ff0': params['fc1'],
        'ff1': params['fc2'],
        'final_layer_norm': params['final_layer_norm'],
    }

def flax2jax(params):
    params = {
        'embedding': {'embedding': params['shared']['embedding']},
        'encoder_embed_positions': params['encoder']['embed_positions']['embedding'],
        'decoder_embed_positions': params['decoder']['embed_positions']['embedding'],
        'encoder_embed_layer_norm': params['encoder']['layernorm_embedding'],
        'decoder_embed_layer_norm': params['decoder']['layernorm_embedding'],
        'encoder_layers': [convert_transformer_encoder(params['encoder']['layers'][str(i)]) for i in range(6)],
        'decoder_layers': [convert_transformer_decoder(params['decoder']['layers'][str(i)]) for i in range(6)],
    }
    return params
