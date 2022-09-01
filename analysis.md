
## Analysis of the Model Architecture

We analyse the inconsistency between the description of the BART model in the literature and the actual BART implementation.

In section 2.1 of the BART paper, the authors stated that BART uses the standard Transformer architecture except for the activation function and initialization. We examine other notable differences between BART and the standard Transformer.

### BART has learnable parameters for Layer Norm

Layer Norm is calculated by this formula:

$$
y = \frac{x-\textbf{E}[x]}{\sqrt{\mathrm{Var}[x]+\epsilon}} * \gamma + \beta
$$

In which $\gamma$ and $\beta$ are learnable parameters.

In section 7 of the Transformer paper, it is said that the Transformer architecture is implemented in the [tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor) library. In the library, [`LayerNormalization`](https://github.com/tensorflow/tensor2tensor/blob/c81d770/tensor2tensor/models/research/residual_shuffle_exchange.py#L40-L41) does not contain learnable parameters.

### BART has extra bias parameters for _Q_, _K_ and _V_

Similar to Layer Norm, BART also has has extra bias parameters for $Q$, $K$ and $V$. It seems that the authors of the BART paper are unaware of this, since they refer $Q$, $K$ and $V$ as 'projection matrix' instead of 'linear layers' in the Section 3.4 of the BART paper.

### Positional encoding is learned rather than fixed

In Section 3.5 of the Transformer paper, it is said that the positional encoding is fixed and is calculated by sine and cosine functions:

$$
PE_{(pos,2i)} = \sin(pos/10000^{2i/d_\mathrm{model}})
$$

$$
PE_{(pos,2i+1)} = \cos(pos/10000^{2i/d_\mathrm{model}})
$$

In BART, however, positional embedding is a learned parameter. The authors of BART seems to be aware of this, since they wrote in Section 3.4 of BART that they were updating the BART positional embeddings in the first training stage of machine translation.

### Positional encoding has an offset of 2

In BART, the positional encoding has an offset of 2, which means that the 0-th token uses the second positional encoding, the first token uses the third positional encoding, an so on. The first two positions of the positional encoding is never used.

### BART uses tied word embeddings

BART uses tied word embeddings on top of the output of the final layer of the decoder. In regular Transformer architecture, the layer is a linear layer used for classification. In BART, however, it is the transpose of the word embedding.

### BART has extra dropout after activation

BART has extra dropout after activation, while Transformer do not have this.

### BART uses an additional decoder start token

TODO: Confirm that Transformer does not use this.

Related: <https://stackoverflow.com/q/64904840>

### BART tokenizer encodes the first word of a sentence differently

```python
from transformers import BartTokenizer

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
inputs = tokenizer(['go go go'], return_tensors='np')
inputs.input_ids.tolist()
```

TODO: Does Chinese BART have this issue?

Related: <https://discuss.huggingface.co/t/bpe-tokenizers-and-spaces-before-words/475>

## Notes on Implementation

This section records the problems we encountered during my implementation of the BART model and the final solutions.

### bart-large has intrinsic problems

This issue is reported in [huggingface/transformers#15559](https://github.com/huggingface/transformers/issues/15559). As a consequence, we only focus on implementing bart-base in this project, and not bart-large.

### `np.std` and `torch.std` are different

```python
import torch

x = torch.tensor([[-1., 1.]])

print(x.std(-1).numpy())  # [1.4142135]
print(x.numpy().std(-1))  # [1.]
```

It is because in `np.std` the denominator is _n_, while in `torch.std` it is _n_-1. See [pytorch/pytorch#1854](https://github.com/pytorch/pytorch/issues/1854) for details.

However, for the standard deviation in Layer Norm, the denominator is always n in either PyTorch or NumPy.

### Computations on TPU are in low precision by default

JAX uses bfloat16 for matrix multiplication on TPU by default, even if the data type is float32. See [google/jax#9973](https://github.com/google/jax/issues/9973) for details.

```python
import jax.numpy as np

print(4176 * 5996)  # 25039296

a = np.array(0.4176, dtype=np.float32)
b = np.array(0.5996, dtype=np.float32)
print((a * b).item())  # 0.25039297342300415
```

For neural network training, however, reducing the accuracy is worthwhile because it can significantly reduce the training time, according to Tom's comments in the above issue.

### Weight matrix of linear layer is transposed in PyTorch

Weight matrix of linear layer is transposed in PyTorch, but not in Flax. Therefore, to convert model parameters between PyTorch and Flax, it is always needed to transpose the weight matrices.

In Flax:

```python
import flax.linen as nn
import jax.numpy as np
import jax.random as rand
linear = nn.Dense(5)
key = rand.PRNGKey(42)
params = linear.init(key, np.zeros((3,)))
print(params['params']['kernel'].shape)  # (3, 5)
```

In PyTorch:

```python
import torch.nn as nn
linear = nn.Linear(3, 5)
print(linear.weight.shape)  # (5, 3), not (3, 5)
```

This can cause sneaky bugs for bart-base, in which the _Q_, _K_, _V_ matrices are square matrices. If the matrices are not transposed, there will be no shape error, but the result will be totally incorrect.

### Layer Norm of PyTorch and Flax are slightly different


### Code quality

TODO: Clean up

If train from scratch, always add `add_prefix_space=True` when initialise the tokenizer

not necessarily true! Cannot have, can cause sneaky bugs. Remember to check it before committing

```
rand.PRNGKey > random.wrapper.seed2key
rand.split > random.wrapper.split_key
rand.KeyArray > lib.random.wrapper.KeyArray
...; del key
```
