from datasets import load_metric
import fire
from transformers import BertTokenizer

bleu = load_metric('bleu')
tokenizer_yue = BertTokenizer.from_pretrained('Ayaka/bart-base-cantonese')

def compute_bleu(predictions_file: str, references_file: str='../wordshk-parallel-corpus/plus15/test.yue.txt', fix_hai: bool=False) -> None:
    with open(predictions_file, encoding='utf-8') as f:
        predictions = [line.rstrip('\n') for line in f]
    with open(references_file, encoding='utf-8') as f:
        references = [line.rstrip('\n') for line in f]

    if fix_hai:
        predictions = [prediction.replace('係', '系') for prediction in predictions]
        references = [reference.replace('係', '系') for reference in references]

    predictions = [tokenizer_yue.tokenize(prediction) for prediction in predictions]
    references = [[tokenizer_yue.tokenize(reference)] for reference in references]

    results = bleu.compute(predictions=predictions, references=references)
    for k, v in results.items():
        print(k, ': ', v, sep='')

if __name__ == '__main__':
    fire.Fire(compute_bleu)
