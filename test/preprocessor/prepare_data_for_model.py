import os; os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_force_host_platform_device_count=2'
import jax; jax.config.update('jax_platforms', 'cpu')
from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from lib.preprocessor.prepare_data_for_model import prepare_data_for_model
from lib.tokeniser import BartTokenizerWithoutOverflowEOS

tokenizer = BartTokenizerWithoutOverflowEOS.from_pretrained('facebook/bart-base')

src, mask_enc_1d = tokenizer(['<mask> go go', 'hi <mask>'], max_length=6)

dst, mask_dec_1d = tokenizer(['go go go', 'hi hi hi hi'], max_length=6)

batch_size, _ = src.shape
assert batch_size % jax.local_device_count() == 0

prepare_data_for_model(src, mask_enc_1d, dst, mask_dec_1d)  # type checking will happen
