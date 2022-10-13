import functools
import jax
import jax.numpy as np
import jax_smi
import optax
import time
import wandb

from lib.dataloader.DataLoader import DataLoader
from lib.model.fwd_transformer import fwd_transformer
from lib.param_utils.init_params import init_params
from lib.param_utils.save_params import save_params
from lib.random.wrapper import seed2key, split_key
from lib.training.cross_entropy_loss import cross_entropy_loss

pad_token_id = 1  # BartTokenizer.from_pretrained('facebook/bart-base').pad_token_id
optimizer = None

@jax.jit
@jax.value_and_grad
def train_forward(params, src, dst, mask_dec_1d, mask_enc, mask_dec, mask_dec_enc, labels, dropout_key):
    outputs = fwd_transformer(params, src, dst, mask_enc, mask_dec, mask_dec_enc, dropout_key=dropout_key)
    lm_head = params['embedding']['embedding'].T
    logits = outputs @ lm_head
    loss = cross_entropy_loss(logits, labels, mask_dec_1d=mask_dec_1d) / len(labels)
    return loss

@functools.partial(jax.pmap, axis_name='n_devices')
def train_step(params, opt_state, src, dst, mask_dec_1d, mask_enc, mask_dec, mask_dec_enc, labels, dropout_key):
    loss, grads = train_forward(params, src, dst, mask_dec_1d, mask_enc, mask_dec, mask_dec_enc, labels, dropout_key=dropout_key)

    grads = jax.lax.psum(grads, axis_name='n_devices')
    loss = jax.lax.psum(loss, axis_name='n_devices')

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss

def main():
    # initialisation

    jax.distributed.initialize()
    jax_smi.initialise_tracking()
    jax.config.update('jax_platforms', 'cpu')  # avoid using TPU in subprocesses
    process_index = jax.process_index()
    if process_index == 0:
        wandb.init(project='bart-pretraining')

    # hyperparameters

    local_devices = jax.local_devices()
    n_local_devices = jax.local_device_count()

    n_epochs = 10
    batch_size_per_device = 80

    key = seed2key(seed=42 + process_index)

    key, subkey = split_key(key)
    data_loader = DataLoader(dataset='enwiki', key=subkey, batch_size_per_device=batch_size_per_device, n_workers=50)

    key, subkey = split_key(key)
    params = init_params(key=subkey)

    global optimizer
    optimizer = optax.lamb(learning_rate=0.0004)
    opt_state = optimizer.init(params)

    replicated_params = jax.device_put_replicated(params, local_devices)
    replicated_opt_state = jax.device_put_replicated(opt_state, local_devices)

    for _ in range(n_epochs):
        if process_index == 0:
            epoch_loss = 0.

        for n_batches, batch in enumerate(data_loader):
            if process_index == 0:
                start_time = time.time()

            key, subkey = split_key(key); subkeys = split_key(subkey, num=n_local_devices)  # force `subkeys` to be an array instead of a list
            replicated_params, replicated_opt_state, replicated_loss = train_step(
                replicated_params,
                replicated_opt_state,
                batch.src,
                batch.dst,
                batch.mask_dec_1d,
                batch.mask_enc,
                batch.mask_dec,
                batch.mask_dec_enc,
                batch.labels,
                dropout_key=subkeys,
            )

            if process_index == 0:
                batch_loss = replicated_loss[0].item()
                assert not np.isnan(batch_loss)
                epoch_loss += batch_loss

                elapsed_time = time.time() - start_time

                wandb.log({'batch loss': batch_loss, 'time': elapsed_time})

        if process_index == 0:
            epoch_loss /= n_batches
            wandb.log({'epoch loss': epoch_loss})

            params = jax.tree_map(lambda x: x[0], replicated_params)
            filename = f'{wandb.run.name}.dat'
            save_params(params, filename)

if __name__ == '__main__':
    main()
