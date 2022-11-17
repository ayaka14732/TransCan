# TransCan: A English-to-Cantonese Machine Translation Model

This project is supported by Cloud TPUs from Google's [TPU Research Cloud](https://sites.research.google/trc/about/) (TRC).

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

However, this common approach is not suitable for Cantonese machine translation. Due to its special writing tradition, which will be discussed below, Cantonese is a low-resource language with only a relatively small English-Cantonese parallel corpus. If a Transformer-based model is trained from scratch on a small corpus, the model will not be fully trained and thus unable to produce meaningful translation results.

One solution to this problem is to exploit an English-to-Mandarin machine translation model. As Cantonese is grammatically similar to Mandarin, which is a high-resource language, the knowledge of Mandarin in an English-to-Mandarin machine translation model would be able to be transferred to Cantonese by fine-tuning the model on a small English-Cantonese parallel dataset. This approach makes the neural machine translation from English to Cantonese feasible. In this research, a model is fine-tuned in this way as a baseline.

However, unlike other low-resource languages, Cantonese is fortunate because it has a relatively large amount of monolingual data. The above approach can only exploit a small bilingual parallel English-Cantonese corpus, while the large monolingual corpus cannot be utilised. This calls for a new approach to be devised to make use of monolingual data.

## My Contributions

- I proposed a novel approach to English-Cantonese machine translation that highly outperforms existing systems;
- For English-to-Cantonese machine translation systems, I proposed the first English-Cantonese machine translation dataset to evaluate their performance;
- For low-resource languages with relatively large mono-lingual data and a pre-trained model for a similar high-resource language, I proposed a novel method to perform a second-stage pre-training to utilise the data and the pre-trained model;
- For low-resource machine translation, I proposed a novel model architecture and fine-tuning method that utilise two monolingual pre-trained models and a small amount of parallel training data to achieve high performance;
- For Chinese NLP tasks, I proposed a method to convert a model trained on a Simplified Chinese dataset to a Traditional Chinese model without any training involved.

## System Overview

The model for machine translation is produced in two steps: pre-training and fine-tuning.

1. Pre-training
    1. **1st-stage pre-training**: Pre-train a Mandarin (simplified Chinese) model _(already been done in [fnlp/bart-base-chinese](https://huggingface.co/fnlp/bart-base-chinese))_
    1. **Embedding conversion**: Convert the embedding to traditional Chinese and add Cantonese vocabulary
    1. **2nd-stage pre-training**: Pre-train the model to fit Cantonese (traditional Chinese) data
2. Fine-tuning
    1. **Architecture modification**: Take the beginning part of the English BART model and the end part of the Cantonese BART model and connect them with two newly-initialised linear projection layers
    1. **1st-stage fine-tuning**: Fine-tune the two newly-initialised linear projection layers
    1. **2nd-stage fine-tuning**: Fine-tune the whole model

## Pre-Training

![](demo/2.png)

TODO: add explanation

Cantonese is grammatically close to Mandarin, and has a relatively large amount of monolingual data. Therefore, it was natural to think that one can pre-train a Mandarin model in the first stage and then further pre-train the model in the second stage to obtain a Cantonese model.

Translation of the above sentences:

- 包括大不列颠岛北部以及**周围**的数个群岛 (meaning: including the northern part of the island of Great Britain and several **surrounding** archipelagos)
- **周圍**嘅朋友都當佢哋係一對 (meaning: All the friends **around me** think they are a couple)
- 佢感冒但係又唔戴口罩**周圍**行 (meaning: He has the flu, but he walks **around** without a mask)

### 1st-Stage Pre-Training

A pre-trained Mandarin BART model has already been proposed in [fnlp/bart-base-chinese](https://huggingface.co/fnlp/bart-base-chinese). Therefore, I have to decide whether to use the pre-trained BART model proposed in that paper or to pre-train a Mandarin BART model from scratch.

There are advantages and disadvantages to both approaches. If I train the BART model from scratch, I can make the necessary modifications to the pre-training data and tokeniser to prepare for the second-stage pre-training. For example, I can use StarCC to convert the pre-training data to traditional Chinese and add Cantonese-specific Chinese characters to the tokeniser in advance. If I utilise the existing BART model, I have to make direct modifications to the embedding, which will be more challenging. However, this would save pre-training resources.

In response to the call for environmental research, and to demonstrate the feasibility of converting a simplified Chinese model to a traditional Chinese model by modifying the embedding, I decided to use the existing pre-trained Mandarin BART model.

### Embedding Conversion

<!-- 首先，tokeniser是在简体中文上训练的，不适合繁体中文文本，因此我需要將tokeniser vocabulary中的簡體字轉為繁體。其次，一些字符只在粵語中使用，需要在tokeniser中添加這些字符。 -->

**Tokeniser:** Convert Simplified Chinese to Traditional Chinese on a word-level.

1. Convert the tokens of the original tokeniser from Simplified Chinese to Traditional Chinese, while keeping the corresponding embeddings fixed
1. Given Cantonese datasets, calculate how many Cantonese-specific characters are missing in the original tokenizer, and add them to the vocabulary
1. Randomly initialise new embeddings for new tokens

(TODO) Elaborate according to [ayaka14732/bert-tokenizer-cantonese](https://github.com/ayaka14732/bert-tokenizer-cantonese).

### 2nd-Stage Pre-Training

**Dataset:** The model is trained on the LIHKG dataset, which consists of 172,937,863 sentences. The average length of the sentence is 17.8 characters. Each sentence is padded or truncated to 64 tokens.

**Pre-training objective**

I adopted the text-infilling objective described in the original BART paper. As the original description is unclear, I propose a robust text-infilling algorithm as follows: Wakong (TODO) (Appendix A)

**Initialisation**: The model is initialised from the Chinese (Mandarin) BART model, with the embedding layer modified as the description above. Extra tokens are randomly initialised. Using weight tying.

**Training details:** I utilise the SGD optimiser with a learning rate of 0.03 and adaptive gradient clipping 0.1, and the batch size is 640. It is trained for 7 epochs and 61,440 steps. The training takes 44.0 hours on Google Cloud TPU v4-16.

## Fine-Tuning

### Architectural Modifications

![](demo/1.png)

The TransCan model is based on the BART base model. The first part of the model is the encoder embedding and all but the last layers of the encoder, which are initialised from the English BART model. The second part of the model is the decoder embedding, the last encoder layer and all layers of the decoder, which are initialised from the Cantonese BART model. Two linear projection layers are inserted between the two parts of the model.

### 1st-Stage Fine-Tuning

(TODO) Add more details

**Initialisation**: In each linear layer, the weights are randomly initialised using Lecun normal initialiser, while the biases are set to zeros.

**Training details:** Only train the two linear projection layers and fix the other parts. The intuition is that ...

### 2nd-Stage Fine-Tuning

(TODO) Add more details

**Training details:** The whole model is fine-tuned. I utilise the AdamW optimiser with a learning rate of 1e-5. Batch size is 32 and trained for 8 epochs.

## Source Code: Repositories

TransCan consists of multiple repositories:

- [ayaka14732/bart-base-jax](https://github.com/ayaka14732/bart-base-jax): Base model architecture
- [ayaka14732/bert-tokenizer-cantonese](https://github.com/ayaka14732/bert-tokenizer-cantonese): Conversion of the embedding
- [ayaka14732/lihkg-scraper](https://github.com/ayaka14732/lihkg-scraper): Data preparation for the 2nd-stage pre-training
- [ayaka14732/wakong](https://github.com/ayaka14732/wakong): Training objective for the 2nd-stage pre-training
- [ayaka14732/bart-base-cantonese](https://github.com/ayaka14732/bart-base-cantonese): Scripts for the 2nd-stage pre-training
- [CanCLID/abc-cantonese-parallel-corpus](https://github.com/CanCLID/abc-cantonese-parallel-corpus): Data for fine-tuning
- [ayaka14732/wordshk-parallel-corpus](https://github.com/ayaka14732/wordshk-parallel-corpus): Data for fine-tuning and evaluation
- [ayaka14732/TransCan](https://github.com/ayaka14732/TransCan): Scripts for fine-tuning and evaluation

Model weights:

- 2nd-stage pre-training: [Hugging Face Hub](https://huggingface.co/Ayaka/bart-base-cantonese)
- 1st-stage fine-tuning: [Google Drive](https://drive.google.com/file/d/1MX0LYW5jhB72g3F_WAKQm1nZVQyuD_nl/view)
- 2nd-stage fine-tuning: [Google Drive](https://drive.google.com/file/d/1IfsLd_KDnYO7nUqN0JcHoy2oLif2u4V6/view)

Training details on W&B:

- 2nd-stage pre-training: [`1j7zs802`](https://wandb.ai/ayaka/bart-base-cantonese/runs/1j7zs802)
- 1st-stage fine-tuning: [`3nqi5cpl`](https://wandb.ai/ayaka/en-kfw-nmt/runs/3nqi5cpl)
- 2nd-stage fine-tuning: [`2ix84gyx`](https://wandb.ai/ayaka/en-kfw-nmt-2nd-stage'/runs/2ix84gyx)

## Source Code: Steps to Reproduce

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
