import os; os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_force_host_platform_device_count=8'
import jax; jax.config.update('jax_platforms', 'cpu')
from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from lib.dataset.load_cantonese import load_cantonese
from lib.preprocessor.Preprocessor import Preprocessor
from lib.random.wrapper import seed2key

if __name__ == '__main__':
    key = seed2key(42)

    sentences = load_cantonese(split='dev')

    preprocessor = Preprocessor(sentences, key=key, batch_size_per_device=6, n_workers=8)
    for n_batches, batch in enumerate(preprocessor):
        print(
            batch.src.shape,
            batch.dst.shape,
            batch.mask_enc_1d.shape,
            batch.mask_dec_1d.shape,
            batch.mask_enc.shape,
            batch.mask_dec.shape,
            batch.mask_dec_enc.shape,
            batch.labels.shape,
        )
        print([buffer.device() for buffer in batch.src.device_buffers])


from lib.tokeniser import EnTokenizer, YueTokenizer

tokenizer_en = EnTokenizer.from_pretrained('facebook/bart-base')
tokenizer_yue = YueTokenizer.from_pretrained('Ayaka/bart-base-cantonese')

batch.src[0][0].tolist()
tokenizer_en.batch_decode(batch.src[0].tolist(), skip_special_tokens=True)

batch.dst[0][0].tolist()
tokenizer_yue.batch_decode(batch.dst[0].tolist(), skip_special_tokens=True)
