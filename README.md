# TransCan: A English-to-Cantonese Machine Translation Model

The model architecture of this project is based on [ayaka14732/bart-base-jax](https://github.com/ayaka14732/bart-base-jax). This project is supported by Cloud TPUs from Google's [TPU Research Cloud](https://sites.research.google/trc/about/) (TRC).

## Inference

To use this model directly, follow the instructions below.

1\. Install Python 3.11

2\. Create a virtual environment (venv)

For example, on Arch Linux, perform the following steps:

```sh
python -m venv venv
. ./venv/bin/activate
```

3\. Update the virtual environment

```sh
pip install -U pip
pip install -U wheel
```

4\. Install JAX and PyTorch

Please refer to the official JAX and PyTorch documentation for installation methods.

- [JAX official installation method](https://github.com/google/jax#installation)
- [PyTorch official installation method](https://pytorch.org/get-started/locally/)

For example, on Arch Linux (with CUDA version 12.1), the commands used for installation are as follows:

```sh
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
```

In fact, running this project does not require a CUDA environment. Therefore, you can simply install the CPU version. Additionally, the generation script defaults to using the CPU and does not utilise CUDA. You need to modify the script to use CUDA.

5\. Install other dependencies

```sh
pip install -r requirements.txt
```

6\. Download the model weights

Download `atomic-thunder-15-7.dat` from [Google Drive](https://drive.google.com/file/d/1IfsLd_KDnYO7nUqN0JcHoy2oLif2u4V6/view).

7\. Run the generation script

```sh
python generate.py atomic-thunder-15-7.dat
```

## Results

| Model | BLEU | File |
| :- | :-: | :- |
| **TransCan** | **28.6** | [`results-bart.txt`](results-bart.txt) |
| Baseline | 21.4 | [`results-baseline.txt`](results-baseline.txt) |
| Baidu Translate | 16.8 | [`results-baidu.txt`](results-baidu.txt) |
| Bing Translate | 15.5 | [`results-bing.txt`](results-bing.txt) |

- Source sentences: [`test.en.txt`](https://github.com/ayaka14732/wordshk-parallel-corpus/blob/v1/plus15/test.en.txt)
- References: [`test.yue.txt`](https://github.com/ayaka14732/wordshk-parallel-corpus/blob/v1/plus15/test.yue.txt)

## Motivation

### Current State of English-to-Cantonese Machine Translation

Cantonese is a language spoken by 85.6 million people worldwide. Both [Baidu Translate](https://fanyi.baidu.com/) and [Bing Translate](https://www.bing.com/translator) have launched their English-to-Cantonese commercial machine translation services. However, the results produced by these two translators are not satisfactory. Although being closed-source and there is no way to view their internal architecture, it is clear that both translators first translate Cantonese into Mandarin, and then utilise a rule-based system to translate Mandarin into Cantonese, by simply taking a look at the translation results:

**Example 1**

- Source: The motor has a broken piston. Let's get a repairman to fix it.
- Reference: 個摩打有條活塞爆咗，揾個師傅嚟整整佢。
- Bing Translate (Mandarin): 電機的活塞損壞。讓我們找一個修理工來修理它。
- Bing Translate (Cantonese): 電機嘅活塞損壞。讓我們找一個修理工來修理它。

In the above example, Bing Translate correctly translated the Mandarin word 的 (*de*, possessive marker) to the Cantonese word 嘅 (*ge3*), but failed to translate the Mandarin word 找 (*zhǎo*, 'find') to Cantonese 揾 (*wan3*) and Mandarin 它 (*tā*, 'it') to Cantonese 佢 (*keoi5*).

**Example 2**

- Source: He has a husky, magnetic voice.
- Reference: 佢把聲沙沙哋、沉沉哋，好有磁性。
- Baidu Translate (Mandarin): 他声音沙哑，富有磁性。
- Baidu Translate (Cantonese): 佢把声沙，有钱磁性。

In the above example, Baidu Translate unexpectedly produced a sentence containing the word 有钱 (*jau5 cin2*, 'rich'), which does not exist in the source sentence. This is because the system translated the word 'magnetic' as 富有磁性 (*fùyǒu cíxìng*, 'full of magnetism'), while the word 富有 (*fùyǒu*) has another meaning of 'rich' in Mandarin.

**Example 3**

- Source: A sugar cube is about the same amount as a teaspoon.
- Reference: 一粒方糖差唔多係一茶匙糖咁滯。
- Baidu Translate (Mandarin): 一块方糖的量大约等于一茶匙。
- Baidu Translate (Cantonese): 一蚊方糖嘅量大约等于一茶匙。

In the above example, Baidu Translate unexpectedly chose the wrong quantifier 蚊 (*man1*) instead of 粒 (*nap1*) to describe a sugar cube. This is because, in the Mandarin translation, the correct quantifier 块 (*kuài*) is used, but the quantifier 块 (*kuài*) corresponds to both 蚊 (*man1*, for money) and 粒 (*nap1*, for sugar cube) in Cantonese.

All the examples above demonstrate that existing commercial English-to-Cantonese machine translation systems produce unsatisfactory results, and even in some cases, the system cannot fully translate Mandarin into Cantonese, resulting in some words being translated into Cantonese and some words remaining in Mandarin, producing an incongruous hybrid of Mandarin and Cantonese. Such a result is offensive to Cantonese speakers and confusing to other users. Therefore, there is an urgent need to improve machine translation from English to Cantonese.

### Current State of Neural Machine Translation

Currently, the most advanced approach to machine translation is neural machine translation, in which Transformer-based models are the most widely adopted model architecture. To train such a model, researchers need to prepare a bilingual parallel corpus from the source language to the target language. As the training objective, the model takes a sentence in the source language as input and outputs the corresponding sentence in the target language.

However, this common approach is not suitable for Cantonese machine translation. Cantonese is a low-resource language with only a relatively small English-Cantonese parallel corpus. If a Transformer-based model is trained from scratch on a small corpus, the model will not be fully trained and thus unable to produce meaningful translation results.

One solution to this problem is to exploit an English-to-Mandarin machine translation model. As Cantonese is grammatically similar to Mandarin, which is a high-resource language, the knowledge of Mandarin in an English-to-Mandarin machine translation model would be able to be transferred to Cantonese by fine-tuning the model on a small English-Cantonese parallel dataset. This approach makes the neural machine translation from English to Cantonese feasible. In this project, a model is fine-tuned in this way as a baseline.

However, unlike other low-resource languages, Cantonese is fortunate because it has a relatively large amount of monolingual data. The above approach can only exploit a small bilingual parallel English-Cantonese corpus, while the large monolingual corpus cannot be utilised. This calls for a new approach to be devised to make use of monolingual data.

## Dataset

### LIHKG Dataset

The [LIHKG dataset](https://github.com/ayaka14732/lihkg-scraper) (denoted by **LIHKG**) is a monolingual Cantonese dataset. The dataset consists of 172,937,863 unique sentences and the average length of the sentences is 17.8 characters.

### ABC Cantonese Parallel Corpus

The [ABC Cantonese Parallel Corpus](https://github.com/CanCLID/abc-cantonese-parallel-corpus) (denoted by **ABC**) is extracted from the [_ABC Cantonese-English Comprehensive Dictionary_](https://wenlin.co/wow/Project:Jyut). The corpus provides 14,474 high-quality Cantonese-English parallel sentences, which is valuable for developing Cantonese-English translation systems.

### Words.hk Cantonese-English Parallel Corpus

The [Words.hk Cantonese-English Parallel Corpus](https://github.com/CanCLID/wordshk-parallel-corpus.git) is extracted from [Words.hk](https://words.hk/), a crowdsourced, sustainably developed, web-based Cantonese dictionary for native speakers and beginners of Cantonese. The dictionary is dedicated to providing complete Cantonese-English bilingual explanations and illustrative examples.

After extracting the parallel data, I split the data into two datasets, according to whether the Cantonese sentences in the Cantonese-English sentence pairs were greater than or equal to 15 characters. The dataset with lengths less than 15 characters are denoted by **Minus15** (29487 pairs of sentences), while the dataset with lengths greater than 15 is denoted by **Plus15** (12372 pairs of sentences).

For the **Plus15** dataset, I further split it into three divisions: train (9372 sentences), dev (1500 sentences) and test (1500 sentences).

I split the *Words.hk* corpus into **Minus15** and **Plus15** according to the length because short sentences are not good metrics for evaluating the performance of machine translation systems. Firstly, short sentences contain less information, so the translation result may not reflect the true ability of a machine translation system. Secondly, BLEU is defined as the geometric mean of _n_-gram precisions (usually for _n_ from 1 to 4), and a brevity penalty to prevent short sentences from receiving unfairly high evaluation scores. For short sentences, the chance of having a matching 4-gram is relatively low, so the reported BLEU scores will be very low and will not be a good indicator of performance. Therefore, it is a better choice to employ long sentences in the eval and test phases of the model training. Separating long sentences from short sentences makes the logic of the subsequent processing easier.

### Utilisation of the Datasets

In this project, the monolingual **LIHKG** dataset is utilised for the second-stage pre-training of the Cantonese BART model. The combination of the **ABC** dataset, the **Minus15** dataset and the train division of the **Plus15** dataset (denoted by **Train**) is utilised for both the first stage and the second stage of the fine-tuning.

## Model

### Baseline Model

For the baseline model, I fine-tuned the OPUS-MT en-zh model in the **Train** dataset. OPUS-MT is a project dedicated to providing open translation services and tools that are free from commercial interests and restrictions.

Although the OPUS-MT en-zh model is said to be trained on a large amount of parallel data from English to many Sinitic languages, including Jin, Mandarin, Gan, Classical Chinese, Hokkien, Wu and Cantonese, the actual content of the dataset is Mandarin-dominant. This means that the model will not be able to obtain a complete knowledge of other Sinitic languages during the pre-training phase. Fortunately, the tokeniser of the model contains enough Cantonese-specific characters such as 噉 (*gam2*), 佢 (*keoi5*) and 咗 (*zo2*), which means that the model is ready to be fine-tuned on the Cantonese dataset without any modification of the embedding.

### My Model

The model for machine translation is produced in two steps: pre-training and fine-tuning.

Pre-training:

- 1st-stage pre-training: Pre-train a Mandarin (simplified Chinese) model (already been done in fnlp/bart-base-chinese);
- Embedding conversion: Convert the embedding to traditional Chinese and add Cantonese vocabulary (see [ayaka14732/bert-tokenizer-cantonese](https://github.com/ayaka14732/bert-tokenizer-cantonese));
- 2nd-stage pre-training: Pre-train the model to fit Cantonese (traditional Chinese) data (see [ayaka14732/bart-base-cantonese](https://github.com/ayaka14732/bart-base-cantonese)).

Fine-tuning:

- Architecture modification: Take the beginning part of the English BART model and the end part of the Cantonese BART model and connect them with two newly-initialised linear projection layers;
- 1st-stage fine-tuning: Fine-tune the two newly-initialised linear projection layers;
- 2nd-stage fine-tuning: Fine-tune the whole model.

![](demo/1.png)

My model is based on the BART base model. The first part of the model is the encoder embedding and all but the last layers of the encoder, which are initialised from the English BART model. The second part of the model is the decoder embedding, the last encoder layer and all layers of the decoder, which are initialised from the Cantonese BART model. Two linear projection layers are inserted between the two parts of the model.

![](demo/2.png)

## Training

### Baseline Model

The baseline model is the Marian model fine-tuned on the **Train** dataset.

I utilised the AdamW optimiser and conducted hyperparameter tuning on the learning rate.

I fine-tuned for at most 10 epochs and utilise an early stopping strategy as follows:

Let $P_i$ be the model parameter after the $i$-th epoch of the fine-tuning ($i \in \left\{ 0, .., 9\right\}$), and $L_i$ the loss of the model on the evaluation dataset with parameter $P_i$. If there exists an $i$ that satisfies $L_i < L_{i+1} < L_{i+2}$, then the model parameter $P_i$ that satisfies this condition with the smallest $i$ is taken as the final result. Otherwise, the parameter of the last epoch is taken as the final result.

### My Model

#### Second-Stage Pre-Training

Each sentence is padded or truncated to 64 tokens.

The training utilises a matrix multiplication precision of BFloat16, although the tensors are always in Float32 format.

For the pre-training objective, I follow the BART and utilise the text-infilling training objective and utilised the [Wakong](https://github.com/ayaka14732/wakong) library.

The model is initialised from the Chinese (Mandarin) BART model, with the embedding layer modified as the description above. Extra tokens are randomly initialised. Weight tying is also used in this model.

I utilise the SGD optimiser with a learning rate of 0.03 and adaptive gradient clipping with parameter 0.1. I also experimented with other optimisers such as Adam and AdamW, but they yield worse results. The batch size is 640. It is trained for 7 epochs and 61,440 steps. The training takes 44.0 hours on Google Cloud TPU v4-16.

#### First-Stage Fine-Tuning

In each linear layer, the weights are randomly initialised using Lecun normal initialiser, while the biases are set to zeros. I only trained the two linear projection layers and fixed the other parts. The training utilises a matrix multiplication precision of BFloat16, although the tensors are always in Float32 format.

#### Second-Stage Fine-Tuning

The model parameters are initialised from the previous checkpoint. The whole model is fine-tuned with the AdamW optimiser with a universal learning rate of 1e-5. The batch size for the training is 32. The model is trained for at most 10 epochs, and the same early stopping strategy in fine-tuning the baseline model is employed.

## Model Weights

- 2nd-stage pre-training: [Hugging Face Hub](https://huggingface.co/Ayaka/bart-base-cantonese)
- 1st-stage fine-tuning: [Google Drive](https://drive.google.com/file/d/1MX0LYW5jhB72g3F_WAKQm1nZVQyuD_nl/view)
- 2nd-stage fine-tuning: [Google Drive](https://drive.google.com/file/d/1IfsLd_KDnYO7nUqN0JcHoy2oLif2u4V6/view)

## Steps to Reproduce

The experiment is conducted on Google Cloud TPU v4-16, and the results can be reproduced with the same setup. Alternatively, the results can be reproduced on any setup with two hosts, each with four default devices (e.g. two hosts with 4 GPU devices on each host). These scripts can also be easily modified to run on other environment setups to produce similar results, but the results would not be exactly the same.

```sh
# Clone source code and datasets
git clone https://github.com/ayaka14732/TransCan.git
git clone https://github.com/CanCLID/abc-cantonese-parallel-corpus.git
git clone https://github.com/CanCLID/wordshk-parallel-corpus.git

# set environment variables
cd TransCan
export EXTERNAL_IP=...  # the IP address of another host, e.g.: 10.130.0.27
echo $EXTERNAL_IP > external-ips.txt

# 1st-stage fine-tuning
git checkout fine-tune-1st-stage
python convert_params.py
./startpod python finetune.py

# 2nd-stage fine-tuning
git checkout fine-tune-2nd-stage
./startpod python 2_finetune.py

# Generate results
python 3_predict.py
python compute_bleu.py results-bart.txt

# Compare with Bing Translator
export TRANSLATE_KEY=...
export ENDPOINT=...
export LOCATION=...
python translate_bing.py
python compute_bleu.py results-bing.txt

# Compare with Baidu Translator
export BAIDU_APP_ID=...
export BAIDU_APP_KEY=...
python translate_baidu.py
python compute_bleu.py results-baidu.txt --fix-hai
```
