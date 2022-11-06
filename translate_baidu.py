import os
import hashlib
import random
import requests
import time
from tqdm import tqdm
from typing import Any

from lib.dataset.load_cantonese import load_cantonese

def chunks(lst: list[Any], chunk_size: int) -> list[list[Any]]:
    '''Yield successive n-sized chunks from lst.'''
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def md5(s: str) -> str:
    return hashlib.md5(s.encode('utf-8')).hexdigest()

app_id = os.environ['BAIDU_APP_ID']
app_key = os.environ['BAIDU_APP_KEY']
api_url = 'https://fanyi-api.baidu.com/api/trans/vip/translate'

def translate(sentences: list[str], src: str, dst: str) -> list[str]:
    sentence = '\n'.join(sentences)

    salt = str(random.randrange(32768, 67108864))
    payload = {
        'q': sentence,
        'from': src,
        'to': dst,
        'appid': app_id,
        'salt': salt,
        'sign': md5(app_id + sentence + salt + app_key),
    }

    request = requests.post(api_url, data=payload)
    request.raise_for_status()
    responses = request.json()
    return [response['dst'] for response in responses['trans_result']]

sentences = load_cantonese(split='test')
sentences_en = [en for en, _ in sentences]
sentences_chunked_en = chunks(sentences_en, 16)

predicts = []

for chunk in tqdm(sentences_chunked_en[84:]):
    predict_hans = translate(chunk, 'en', 'yue')
    time.sleep(1.)
    predict_hant = translate(predict_hans, 'zh', 'cht')
    time.sleep(1.)
    predicts.extend(predict_hant)

with open('results-baidu.txt', 'a', encoding='utf-8') as f:
    for predict in predicts:
        print(predict, file=f)
