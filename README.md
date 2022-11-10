# TransCan: A Novel Approach to English-Cantonese Machine Translation

**Abstract:** In this paper, I propose TransCan, a novel BART-based approach to English-Cantonese machine translation. Cantonese is a low-resource language with limited English-Cantonese parallel data, which makes the English-Cantonese translation a challenging task. However, Cantonese is grammatically closer to the high-resource Mandarin Chinese language and has a relatively large amount of monolingual data. To exploit these two features of Cantonese, I first perform a second-stage pre-training of the Chinese BART model on a monolingual Cantonese corpus to obtain a Cantonese BART model. Consequently, I devised a simple yet novel model architecture, linking the first half of the English BART model and the second half of the Cantonese BART model with two simple linear projection layers, and fine-tuning the model on a small English-Cantonese parallel corpus. The resulting model outperformed the state-of-the-art commercial machine translation product by 11.8 BLEU. The source code is publicly available on GitHub.

My main contributions are as follows:

1. For English-Cantonese machine translation systems, I proposed the first English-Cantonese machine translation dataset to evaluate their performance
1. For low-resource languages with relatively large mono-lingual data, I proposed a method to perform a second-stage pre-training to utilise the pre-trained models for a similar high-resource language
1. For low-resource machine translation, I proposed a novel model architecture and fine-tuning method that utilise two monolingual pre-trained models and a small amount of parallel training data to obtain a high performance
1. For Chinese NLP tasks, I proposed a method to convert a model trained on a Simplified Chinese dataset to a Traditional Chinese model without any training involved

This research is supported by Cloud TPUs from Google's [TPU Research Cloud](https://sites.research.google/trc/about/) (TRC).

## Results

| Model | BLEU | File |
| :- | :-: | :- |
| **This Model** | **0.286** | [`results-bart.txt`](results-bart.txt) |
| Baidu Translate | 0.168 | [`results-baidu.txt`](results-baidu.txt) |
| Bing Translate | 0.155 | [`results-bing.txt`](results-bing.txt) |

- Source sentences: [`test.en.txt`](https://github.com/ayaka14732/wordshk-parallel-corpus/blob/v1/plus15/test.en.txt)
- References: [`test.yue.txt`](https://github.com/ayaka14732/wordshk-parallel-corpus/blob/v1/plus15/test.yue.txt)

## Repositories

TransCan consists of multiple repositories:

- [ayaka14732/lihkg-scraper](https://github.com/ayaka14732/lihkg-scraper): Data preparation for the 2nd-stage pre-training
- [ayaka14732/wordshk-parallel-corpus](https://github.com/ayaka14732/wordshk-parallel-corpus): Data for fine-tuning and evaluation
- [CanCLID/abc-cantonese-parallel-corpus](https://github.com/CanCLID/abc-cantonese-parallel-corpus): Data for fine-tuning
- [ayaka14732/wakong](https://github.com/ayaka14732/wakong): Training objective for the 2nd-stage pre-training
- [ayaka14732/bert-tokenizer-cantonese](https://github.com/ayaka14732/bert-tokenizer-cantonese): BERT Tokenizer with vocabulary tailored for Cantonese
- [ayaka14732/bart-base-jax](https://github.com/ayaka14732/bart-base-jax): The base model for all experiments in this research
- [ayaka14732/bart-base-cantonese](https://github.com/ayaka14732/bart-base-cantonese): Scripts for the 2nd-stage pre-training
- [ayaka14732/TransCan](https://github.com/ayaka14732/TransCan): Scripts for fine-tuning and evaluation

Model weights:

- 2nd-stage pre-training: [Hugging Face Hub](https://huggingface.co/Ayaka/bart-base-cantonese)
- 1st-stage fine-tuning: [Google Drive](https://drive.google.com/file/d/1MX0LYW5jhB72g3F_WAKQm1nZVQyuD_nl/view)
- 2nd-stage fine-tuning: [Google Drive](https://drive.google.com/file/d/1IfsLd_KDnYO7nUqN0JcHoy2oLif2u4V6/view)

Training details on W&B:

- 2nd-stage pre-training: [`1j7zs802`](https://wandb.ai/ayaka/bart-base-cantonese/runs/1j7zs802)
- 1st-stage fine-tuning: [`3nqi5cpl`](https://wandb.ai/ayaka/en-kfw-nmt/runs/3nqi5cpl)
- 2nd-stage fine-tuning: [`2ix84gyx`](https://wandb.ai/ayaka/en-kfw-nmt-2nd-stage'/runs/2ix84gyx)

## 2nd-Stage Pre-Training

**Tokeniser:** Convert Simplified Chinese to Traditional Chinese on a word-level.

(TODO) Elaborate according to [ayaka14732/bert-tokenizer-cantonese](https://github.com/ayaka14732/bert-tokenizer-cantonese).

(TODO) use an image to illustrate this:

1. Random initialise the embedding
1. 1st-stage pre-training: ……包括大不列颠岛北部以及**周围**的数个群岛 (meaning: ... included the northern part of the island of Great Britain and several **surrounding** archipelagos)
1. token mapping: 周 (keep), 围 -> 圍 (convert SC to TC)
1. 2nd-stage pre-training: 佢感冒但係又唔戴口罩**周圍**行 (meaning: He has the flu, but he goes around without a mask)
1. The final embedding is produced

**Dataset:** The model is trained on the LIHKG dataset, which consists of 172,937,863 sentences. The average length of the sentence is 17.8 characters. Each sentence is padded or truncated to 64 tokens.

**Pre-training objective:** I adopted the text-infilling objective described in the original BART paper. As the original description is unclear, I propose a robust text-infilling algorithm as follows: (TODO)

**Initialisation**: The model is initialised from the Chinese (Mandarin) BART model, with the embedding layer modified as the description above. Extra tokens are randomly initialised. Using weight tying.

**Training details:** I utilise the SGD optimiser with a learning rate of 0.03 and adaptive gradient clipping 0.1, and the batch size is 640. It is trained for 7 epochs and 61,440 steps. The training takes 44.0 hours on Google Cloud TPU v4-16.

## 1st-Stage Fine-Tuning

(TODO) Add more details

**Model Architecture:** The TransCan model is based on the BART base model. The first part of the model is the encoder embedding and all but the last layers of the encoder, which are initialised from the English BART model. The second part of the model is the decoder embedding, the last encoder layer and all layers of the decoder, which are initialised from the Cantonese BART model. Two linear projection layers are inserted between the two parts of the model.

![](demo/1.png)

**Initialisation**: In each linear layer, the weights are randomly initialised using Lecun normal initialiser, while the biases are set to zeros.

**Training details:** Only train the two linear projection layers and fix the other parts. The intuition is that ...

## 2nd-Stage Fine-Tuning

(TODO) Add more details

**Training details:** The whole model is fine-tuned. I utilise the AdamW optimiser with a learning rate of 1e-5. Batch size is 32 and trained for 8 epochs.

## Steps to reproduce

The experiment is carried out on Google Cloud TPU v4-16, and the results can be reproduced on the same setups. Alternatively, the results can be reproduced on any setup with two hosts, each with four default devices (e.g. two hosts with 4 GPU devices on each host). These scripts can also be run on other environment setups to produce similar results, but the results would not be exactly the same.

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
