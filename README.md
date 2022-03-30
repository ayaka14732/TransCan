# JAX Implementation of bart-base

* [1. Motivation](#1-motivation)
* [2. Model Architecture](#2-model-architecture)
    * [2.1. Layer Norm](#21-layer-norm)
    * [2.2. Embedding](#22-embedding)
    * [2.3. Linear](#23-linear)
    * [2.4. Attention](#24-attention)
    * [2.5. Transformer Encoder](#25-transformer-encoder)
    * [2.6. Transformer Decoder](#26-transformer-decoder)
    * [2.7. Transformer](#27-transformer)
* [3. Model Parameters](#3-model-parameters)
    * [3.1. Parameter format of FlaxBartForSequenceClassification](#31-parameter-format-of-flaxbartforsequenceclassification)
    * [3.2. Parameter format of this project](#32-parameter-format-of-this-project)
* [4. Training Setup](#4-training-setup)
* [5. Implementation Notes](#5-implementation-notes)
    * [5.1. The bart-large model itself does not work properly](#51-the-bart-large-model-itself-does-not-work-properly)
    * [5.2. np.std and torch.std are different](#52-npstd-and-torchstd-are-different)
    * [5.3. Computations on TPU are in low precision by default](#53-computations-on-tpu-are-in-low-precision-by-default)
    * [5.4. BART has extra bias parameters for Layer Norm](#54-bart-has-extra-bias-parameters-for-layer-norm)
    * [5.5. BART has extra bias parameters for <em>Q</em>, <em>K</em> and <em>V</em>](#55-bart-has-extra-bias-parameters-for-q-k-and-v)
    * [5.6. Positional Encoding is learned rather than fixed](#56-positional-encoding-is-learned-rather-than-fixed)
    * [5.7. BART uses tied word embeddings](#57-bart-uses-tied-word-embeddings)
    * [5.8. Hugging Face Transformers 4.17.0 is not compactible with JAX 0.3.4](#58-hugging-face-transformers-4170-is-not-compactible-with-jax-034)

<!-- Created by https://github.com/ekalinin/github-markdown-toc -->

## 1. Motivation

This project is the JAX implementation of the [bart-base](https://arxiv.org/abs/1910.13461) model. It is built with two objectives in mind:

(1) To explain the BART architecture more clearly;

(2) To demonstrate how the Transformer-based model can be implemented in JAX.

In addition to the regular implementation, I also implemented the model [in a single line of Python code](https://twitter.com/ayaka14732/status/1507955631109869574), by virtue of JAX's functional-style API.

This project is inspired by [hyunwoongko/transformer](https://github.com/hyunwoongko/transformer). Nevertheless, the code is written entirely on my own.

## 2. Model Architecture

### 2.1. Layer Norm

![](https://raw.githubusercontent.com/hyunwoongko/transformer/master/image/layer_norm.jpg)

```python
def fwd_layer_norm(params: dict, x: np.ndarray, eps: float=1e-5) -> np.ndarray:
    # params
    scale: np.ndarray = params['scale']  # array
    bias: np.ndarray = params['bias']  # array

    mean = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps)) * scale + bias
```

### 2.2. Embedding

```python
def fwd_embedding(params: dict, x: np.ndarray) -> np.ndarray:
    # params
    embedding: np.ndarray = params['embedding']  # array

    y = embedding[x]
    return y
```

### 2.3. Linear

```python
def fwd_linear(params: dict, x: np.ndarray) -> np.ndarray:
    # params
    kernel: np.ndarray = params['kernel']  # array
    bias: np.ndarray = params['bias']  # array

    return np.dot(x, kernel) + bias
```

### 2.4. Attention

![](https://raw.githubusercontent.com/hyunwoongko/transformer/master/image/multi_head_attention.jpg)

```python
def fwd_attention(params: dict, src: np.ndarray, dst: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # params
    q_proj: dict = params['q_proj']  # linear
    k_proj: dict = params['k_proj']  # linear
    v_proj: dict = params['v_proj']  # linear
    ff: dict = params['ff']  # linear

    _, _, d_k = q_proj['kernel'].shape

    q = fwd_linear(q_proj, dst)
    k = fwd_linear(k_proj, src)
    v = fwd_linear(v_proj, src)

    qk = np.einsum('bkhm,bvhm->bhkv', q, k)
    qk = qk / np.sqrt(d_k)
    qk = np.where(mask, qk, np.NINF)
    qk = nn.softmax(qk)
    qk = np.where(mask, qk, 0)

    t = np.einsum('bhkv,bvhm->bkhm', qk, v)
    d0, d1, d2, d3 = t.shape
    t = t.reshape(d0, d1, d2 * d3)

    t = fwd_linear(ff, t)
    return t
```

### 2.5. Transformer Encoder

```python
def fwd_transformer_encoder(params: dict, src: np.ndarray, mask_enc: np.ndarray) -> np.ndarray:
    # params
    self_attn: dict = params['self_attn']  # attention
    self_attn_layer_norm: dict = params['self_attn_layer_norm']  # layer norm
    ff0: dict = params['ff0']  # linear
    ff1: dict = params['ff1']  # linear
    final_layer_norm: dict = params['final_layer_norm']  # layer norm

    src_ = src
    t = fwd_attention(self_attn, src, src, mask_enc)
    t = t + src_
    t = fwd_layer_norm(self_attn_layer_norm, t)

    t_ = t
    t = fwd_linear(ff0, t)
    t = nn.gelu(t)
    t = fwd_linear(ff1, t)
    t = t + t_
    t = fwd_layer_norm(final_layer_norm, t)
    return t
```

### 2.6. Transformer Decoder

```python
def fwd_transformer_decoder(params: dict, src: np.ndarray, dst: np.ndarray, mask_dec: np.ndarray, mask_dec_enc: np.ndarray) -> np.ndarray:
    # params
    self_attn: dict = params['self_attn']  # attention
    self_attn_layer_norm: dict = params['self_attn_layer_norm']  # layer norm
    cross_attn: dict = params['cross_attn']  # attention
    cross_attn_layer_norm: dict = params['cross_attn_layer_norm']  # layer norm
    ff0: dict = params['ff0']  # linear
    ff1: dict = params['ff1']  # linear
    final_layer_norm: dict = params['final_layer_norm']  # layer norm

    dst_ = dst
    dst = fwd_attention(self_attn, dst, dst, mask_dec)
    dst = dst + dst_
    dst = fwd_layer_norm(self_attn_layer_norm, dst)

    dst_ = dst
    src = fwd_attention(cross_attn, src, dst, mask_dec_enc)
    t = src + dst_
    t = fwd_layer_norm(cross_attn_layer_norm, t)

    t_ = t
    t = fwd_linear(ff0, t)
    t = nn.gelu(t)
    t = fwd_linear(ff1, t)
    t = t + t_
    t = fwd_layer_norm(final_layer_norm, t)
    return t
```

### 2.7. Transformer

```python
def fwd_transformer(params: dict, src: np.ndarray, dst: np.ndarray, mask_enc: np.ndarray, mask_dec: np.ndarray, mask_dec_enc: np.ndarray) -> np.ndarray:
    # params
    embedding: dict = params['embedding']  # embedding
    encoder_embed_positions: np.ndarray = params['encoder_embed_positions']  # array
    decoder_embed_positions: np.ndarray = params['decoder_embed_positions']  # array
    encoder_embed_layer_norm: dict = params['encoder_embed_layer_norm']  # layer norm
    decoder_embed_layer_norm: dict = params['decoder_embed_layer_norm']  # layer norm
    encoder_layers: list = params['encoder_layers']  # list of transformer encoder
    decoder_layers: list = params['decoder_layers']  # list of transformer encoder

    _, width_enc = src.shape
    _, width_dec = dst.shape

    # https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/bart/modeling_flax_bart.py#L718-L719
    offset = 2

    src = fwd_embedding(embedding, src)
    src = src + encoder_embed_positions[offset:width_enc+offset]
    src = fwd_layer_norm(encoder_embed_layer_norm, src)
    for encoder_layer in encoder_layers:
        src = fwd_transformer_encoder(encoder_layer, src, mask_enc)

    dst = fwd_embedding(embedding, dst)
    dst = dst + decoder_embed_positions[offset:width_dec+offset]
    dst = fwd_layer_norm(decoder_embed_layer_norm, dst)
    for decoder_layer in decoder_layers:
        dst = fwd_transformer_decoder(decoder_layer, src, dst, mask_dec, mask_dec_enc)

    return dst
```

## 3. Model Parameters

### 3.1. Parameter format of `FlaxBartForSequenceClassification`

```
shared
    embedding (50265, 768)
encoder
    embed_positions
        embedding (1026, 768)
    layernorm_embedding
        scale (768,)
        bias (768,)
    layers
        0..5
            self_attn
                q_proj
                    kernel (768, 768)
                    bias (768,)
                k_proj
                    kernel (768, 768)
                    bias (768,)
                v_proj
                    kernel (768, 768)
                    bias (768,)
                out_proj
                    kernel (768, 768)
                    bias (768,)
            self_attn_layer_norm
                scale (768,)
                bias (768,)
            fc1
                kernel (768, 3072)
                bias (3072,)
            fc2
                kernel (3072, 768)
                bias (768,)
            final_layer_norm
                scale (768,)
                bias (768,)
decoder
    embed_positions
        embedding (1026, 768)
    layernorm_embedding
        scale (768,)
        bias (768,)
    layers
        0..5
            self_attn
                q_proj
                    kernel (768, 768)
                    bias (768,)
                k_proj
                    kernel (768, 768)
                    bias (768,)
                v_proj
                    kernel (768, 768)
                    bias (768,)
                out_proj
                    kernel (768, 768)
                    bias (768,)
            self_attn_layer_norm
                scale (768,)
                bias (768,)
            encoder_attn
                q_proj
                    kernel (768, 768)
                    bias (768,)
                k_proj
                    kernel (768, 768)
                    bias (768,)
                v_proj
                    kernel (768, 768)
                    bias (768,)
                out_proj
                    kernel (768, 768)
                    bias (768,)
            encoder_attn_layer_norm
                scale (768,)
                bias (768,)
            fc1
                kernel (768, 3072)
                bias (3072,)
            fc2
                kernel (3072, 768)
                bias (768,)
            final_layer_norm
                scale (768,)
                bias (768,)
```

### 3.2. Parameter format of this project

```
embedding
    embedding (50265, 768)
encoder_embed_positions (1026, 768)
decoder_embed_positions (1026, 768)
encoder_embed_layer_norm
    scale (768,)
    bias (768,)
decoder_embed_layer_norm
    scale (768,)
    bias (768,)
encoder_layers
    0..5
        self_attn
            q_proj
                kernel (12, 768, 64)
                bias (12, 64)
            k_proj
                kernel (12, 768, 64)
                bias (12, 64)
            v_proj
                kernel (12, 768, 64)
                bias (12, 64)
            ff
                kernel (768, 768)
                bias (768,)
        self_attn_layer_norm
            scale (768,)
            bias (768,)
        ff0
            kernel (768, 3072)
            bias (3072,)
        ff1
            kernel (3072, 768)
            bias (768,)
        final_layer_norm
            scale (768,)
            bias (768,)
decoder_layers
    0..5
        self_attn
            q_proj
                kernel (12, 768, 64)
                bias (12, 64)
            k_proj
                kernel (12, 768, 64)
                bias (12, 64)
            v_proj
                kernel (12, 768, 64)
                bias (12, 64)
            ff
                kernel (768, 768)
                bias (768,)
        self_attn_layer_norm
            scale (768,)
            bias (768,)
        cross_attn
            q_proj
                kernel (12, 768, 64)
                bias (12, 64)
            k_proj
                kernel (12, 768, 64)
                bias (12, 64)
            v_proj
                kernel (12, 768, 64)
                bias (12, 64)
            ff
                kernel (768, 768)
                bias (768,)
        cross_attn_layer_norm
            scale (768,)
            bias (768,)
        ff0
            kernel (768, 3072)
            bias (3072,)
        ff1
            kernel (3072, 768)
            bias (768,)
        final_layer_norm
            scale (768,)
            bias (768,)
```

## 4. Training Setup

## 5. Implementation Notes

This section records the problems I encountered during my implementation of the BART model and the final solutions.

### 5.1. The bart-large model itself does not work properly

This issue is reported in [huggingface/transformers#15559](https://github.com/huggingface/transformers/issues/15559). As a consequence, I only focus on implementing bart-base in this project, and not bart-large.

### 5.2. `np.std` and `torch.std` are different

```python
import torch

x = torch.tensor([[-1., 1.]])

print(x.std(-1).numpy())  # [1.4142135]
print(x.numpy().std(-1))  # [1.]
```

It is because in `np.std` the denominator is _n_, while in `torch.std` it is _n_-1. See [pytorch/pytorch#1854](https://github.com/pytorch/pytorch/issues/1854) for details.

However, for the standard deviation in Layer Norm, the denominator is always n in either PyTorch or NumPy.

### 5.3. Computations on TPU are in low precision by default

JAX uses bfloat16 for matrix multiplication on TPU by default, even if the data type is float32. See [google/jax#9973](https://github.com/google/jax/issues/9973) for details.

```python
import jax.numpy as np

print(4176 * 5996)  # 25039296

a = np.array(0.4176, dtype=np.float32)
b = np.array(0.5996, dtype=np.float32)
print((a * b).item())  # 0.25039297342300415
```

For neural network training, however, reducing the accuracy is worthwhile because it can significantly reduce the training time, according to Tom's comments in the above issue.

### 5.4. BART has extra bias parameters for Layer Norm

In section 2.1 of the BART paper, it is stated that BART uses the standard Transformer architecture, except for the activation function and initialization. However, this is not true because BART has extra bias parameters for Layer Norm.

TODO: Add the formula of Layer Norm here.

TODO: Add a proof that the original Transformer architecture does not have bias for Layer Norm.

### 5.5. BART has extra bias parameters for _Q_, _K_ and _V_

Besides Layer Norm, BART also has has extra bias parameters for _Q_, _K_ and _V_.

TODO: Add demonstration.

### 5.6. Positional Encoding is learned rather than fixed

### 5.7. BART uses tied word embeddings

### 5.8. Hugging Face Transformers 4.17.0 is not compactible with JAX 0.3.4
