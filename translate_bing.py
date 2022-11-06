import os
import requests
from tqdm import tqdm
from typing import Any
import uuid

from lib.dataset.load_cantonese import load_cantonese

def chunks(lst: list[Any], chunk_size: int) -> list[list[Any]]:
    '''Yield successive n-sized chunks from lst.'''
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

key = os.environ['TRANSLATE_KEY']
endpoint = os.environ['ENDPOINT']
location = os.environ['LOCATION']

path = '/translate'
constructed_url = endpoint + path
params = {
    'api-version': '3.0',
    'from': 'en',
    'to': ['yue']
}
headers = {
    'Ocp-Apim-Subscription-Key': key,
    'Ocp-Apim-Subscription-Region': location,
    'Content-type': 'application/json',
    'X-ClientTraceId': str(uuid.uuid4())
}

sentences = load_cantonese(split='test')
sentences_en = [en for en, _ in sentences]
sentences_chunked_en = chunks(sentences_en, 32)

predicts = []

for chunk in tqdm(sentences_chunked_en):
    body = [{'text': sentence} for sentence in chunk]
    request = requests.post(constructed_url, params=params, headers=headers, json=body)
    responses = request.json()
    for predict in responses:
        predicts.append(predict['translations'][0]['text'])

with open('results-bing.txt', 'w', encoding='utf-8') as f:
    for predict in predicts:
        print(predict, file=f)
