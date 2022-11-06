# English-Cantonese Translation Model

This project is supported by Cloud TPUs from Google's [TPU Research Cloud](https://sites.research.google/trc/about/) (TRC).

## Results

| Model | BLEU |
| :- | :-: |
| **This Model** | **0.286** |
| Baidu Translate | 0.168 |
| Bing Translate | 0.155 |

## Develop

```sh
# Clone source code and datasets
git clone https://github.com/ayaka14732/bart-base-jax.git
git clone https://github.com/CanCLID/abc-cantonese-parallel-corpus.git
git clone https://github.com/CanCLID/wordshk-parallel-corpus.git

# 1st-stage fine-tuning
cd bart-base-jax
git checkout en-kfw-nmt
python 1_convert_bart_params.py
./startpod python 2_finetune.py

# 2nd-stage fine-tuning
git checkout en-kfw-nmt-2nd-stage
JAX_PLATFORMS='' python 2_finetune.py

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
