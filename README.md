# JAX Implementation of bart-base

This project is a JAX implementation of the [bart-base](https://arxiv.org/abs/1910.13461) model. The aim of this project is to demonstrate how Transformer-based language models can be implemented using JAX and trained on Cloud TPUs.

This project is inspired by [hyunwoongko/transformer](https://github.com/hyunwoongko/transformer), while the code for this project is entirely written by myself.

## News

**2022-03-27:** In addition to the regular implementation, I also implemented the model in a single line of Python code, by virtue of JAX's functional-style API. [[Twitter]](https://twitter.com/ayaka14732/status/1507955631109869574)

## Environment Setup

Set up TPU environment as described in [ayaka14732/tpu-starter](https://github.com/ayaka14732/tpu-starter). Then run the following commands:

```sh
python3.10 -m venv ./venv
. ./venv/bin/activate
pip install -U pip
pip install -U wheel
pip install "jax[tpu]==0.3.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -r requirements.txt
```

To setup in CUDA environment, simply change `[tpu]` in the above command to `[cuda]`.

## Model Architecture

TODO: See `lib/model`.

## Model Parameters

Parameter-related operations are implemented in the `lib/param_utils` directory. Notably, three functions, `flax2jax`, `pt2jax` and `jax2flax` are implemented, to allow any conversions between PyTorch, Flax and JAX implementation.

| from\to | PyTorch | Flax | JAX |
| :- | :-: | :-: | :-: |
| PyTorch | - | `save_pretrained` | `pt2jax` |
| Flax | `save_pretrained` | - | `flax2jax` |
| JAX | `jax2flax` + `save_pretrained` | `jax2flax` | - |

`save_pretrained` is a function provided by the Hugging Face Transformers library, so that users can save the model in one framework and reload it in another framework. For instance, the following code saves a Flax model and reload it as a PyTorch model:

```python
with tempfile.TemporaryDirectory() as tmpdirname:
    model_flax.save_pretrained(tmpdirname)
    model_pt = BartForConditionalGeneration.from_pretrained(tmpdirname, from_flax=True)
```

JAX parameters, see [param_format.txt](param_format.txt).

## Dataset

### English Wikipedia

Split English Wikipedia into sentences.

1. Download the English Wikipedia data
1. Extract the data by WikiExtractor
1. Split the articles into sentences by Bling Fire
1. Save the sentences to files (one sentence per line)

```sh
python prepare_dataset.py
```

On Cloud TPU v3-8, the processing takes 15m18s. On Cloud TPU v4-8, it takes 4m19s. The size of the directory is about 15 GiB.

## Data Preprocessing

The `[EOS]` token (`tokenizer.eos_token_id`) should be prepended before all sentences in `dst`.

Example:

```
Input: ['<s>A flower.</s><pad>', '<s>Some good sentences.</s>']
Output: ['</s><s>A flower.</s><pad>', '</s><s>Some good sentences.</s>']
Input IDs: [[0, 250, 14214, 4, 2, 1], [0, 6323, 205, 11305, 4, 2]]
Output IDs: [[2, 0, 250, 14214, 4, 2, 1], [2, 0, 6323, 205, 11305, 4, 2]]
```

- **src**: `[BOS]`, A, `[MSK]`, flower, `[EOS]`, `[PAD]`, `[PAD]`
- **dst**: `[EOS]`, `[BOS]`, A, beautiful, flower, `[EOS]`, `[PAD]`
- **label**: `[BOS]`, A, beautiful, flower, `[EOS]`, `[PAD]`, `[PAD]`

<details>

```python
from transformers import BartTokenizer, BartForConditionalGeneration

model_name = 'facebook/bart-base'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

sentences = ('A flower.', 'Some good sentences.')

inputs = tokenizer(sentences, return_tensors='pt', max_length=6, padding='max_length', truncation=True)
output = model.generate(inputs.input_ids)

print('Input:', tokenizer.batch_decode(inputs.input_ids))
print('Output:', tokenizer.batch_decode(output))

print('Input IDs:', inputs.input_ids.tolist())
print('Output IDs:', output.tolist())
```

</details>

## Data Loader

On-demand data loader

## Training

## Evaluation

## Generation

Typical generation process of the BART model involves the input sequences and their masks. The model generates the output autoregressively.

While greedy decoding is the simplest generation algorithm for autoregressive language models, other algorithms like beam search and sampling can improve the quality of the generated sentences and therefore improve performance. In this project, we refrain from implementing these generation algorithms and leave the work to the Hugging Face Transformers library.

However, generation functions in the Hugging Face Transformers library are coupled with the implementation of their original models, which makes them inaccessible for customized models. To tackle this problem, we convert our model to a regular Hugging Face Transformer model.

### Code quality

TODO: Clean up

If train from scratch, always add `add_prefix_space=True` when initialise the tokenizer

Cannot have

can cause sneaky bugs. Remember to check it before committing

not necessarily true!

```
rand.PRNGKey > random.wrapper.seed2key
rand.split > random.wrapper.split_key
rand.KeyArray > lib.random.wrapper.KeyArray
42
...; del key
```

`~/.cache/dump2`
