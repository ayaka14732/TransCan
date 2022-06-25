import jax
import jax.numpy as np

def dotted_dict2nested_dict(params):
    for k in list(params):
        if '.' in k:
            ks = k.split('.')
            p = params
            for nk in ks[:-1]:
                if nk not in p:
                    p[nk] = {}
                p = p[nk]
            final_k = ks[-1]
            p[final_k] = params[k]
            params.pop(k)  # TODO: rewrite in functional API
    return params

def convert_qkv(params):
    return {
        'kernel': params['weight'].T.reshape(768, 12, 64),
        'bias': params['bias'].reshape(12, 64),
    }

def convert_ff(params):
    return {
        'kernel': params['weight'].T.reshape(12, 64, 768),
        'bias': params['bias'],
    }

def convert_linear(params):
    return {
        'kernel': params['weight'].T,
        'bias': params['bias'],
    }

def convert_layer_norm(params):
    return {
        'scale': params['weight'],
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
        'self_attn_layer_norm': convert_layer_norm(params['self_attn_layer_norm']),
        'ff0': convert_linear(params['fc1']),
        'ff1': convert_linear(params['fc2']),
        'final_layer_norm': convert_layer_norm(params['final_layer_norm']),
    }

def convert_transformer_decoder(params):
    return {
        'self_attn': {
            'q_proj': convert_qkv(params['self_attn']['q_proj']),
            'k_proj': convert_qkv(params['self_attn']['k_proj']),
            'v_proj': convert_qkv(params['self_attn']['v_proj']),
            'ff': convert_ff(params['self_attn']['out_proj']),
        },
        'self_attn_layer_norm': convert_layer_norm(params['self_attn_layer_norm']),
        'cross_attn': {
            'q_proj': convert_qkv(params['encoder_attn']['q_proj']),
            'k_proj': convert_qkv(params['encoder_attn']['k_proj']),
            'v_proj': convert_qkv(params['encoder_attn']['v_proj']),
            'ff': convert_ff(params['encoder_attn']['out_proj']),
        },
        'cross_attn_layer_norm': convert_layer_norm(params['encoder_attn_layer_norm']),
        'ff0': convert_linear(params['fc1']),
        'ff1': convert_linear(params['fc2']),
        'final_layer_norm': convert_layer_norm(params['final_layer_norm']),
    }

def pt2jax(params):
    params = dotted_dict2nested_dict(params)
    params = jax.tree_map(lambda x: np.asarray(x.detach().numpy()), params)
    params = {
        'embedding': {'embedding': params['shared']['weight']},
        'encoder_embed_positions': params['encoder']['embed_positions']['weight'],
        'decoder_embed_positions': params['decoder']['embed_positions']['weight'],
        'encoder_embed_layer_norm': convert_layer_norm(params['encoder']['layernorm_embedding']),
        'decoder_embed_layer_norm': convert_layer_norm(params['decoder']['layernorm_embedding']),
        'encoder_layers': [convert_transformer_encoder(params['encoder']['layers'][str(i)]) for i in range(6)],
        'decoder_layers': [convert_transformer_decoder(params['decoder']['layers'][str(i)]) for i in range(6)],
    }
    return params
