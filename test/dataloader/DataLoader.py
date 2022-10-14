import os; os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_force_host_platform_device_count=8'
import jax; jax.config.update('jax_platforms', 'cpu')
from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from lib.dataloader.DataLoader import DataLoader
from lib.random.wrapper import seed2key

if __name__ == '__main__':
    key = seed2key(42)

    data_loader = DataLoader(dataset='dummy', key=key, batch_size_per_device=6, n_workers=32)
    for n_batches, batch in enumerate(data_loader):
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
