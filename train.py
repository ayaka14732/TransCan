import functools
import jax
import jax_smi
import optax
import time
import wandb

from lib.dataset.enwiki import load_enwiki
from lib.model import fwd_transformer
from lib.param_utils.init_params import init_params
from lib.param_utils.save_params import save_params
from lib.preprocessor.Preprocessor import Preprocessor
from lib.random.wrapper import seed2key, split_key
from lib.training.cross_entropy_loss import cross_entropy_loss

pad_token_id = 1  # BartTokenizerWithoutOverflowEOS.from_pretrained('facebook/bart-base').pad_token_id
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

@functools.partial(jax.pmap, axis_name='n_devices')
def eval_step(params, src, dst, mask_dec_1d, mask_enc, mask_dec, mask_dec_enc, labels):
    outputs = fwd_transformer(params, src, dst, mask_enc, mask_dec, mask_dec_enc)
    lm_head = params['embedding']['embedding'].T
    logits = outputs @ lm_head
    loss = cross_entropy_loss(logits, labels, mask_dec_1d=mask_dec_1d) / len(labels)
    return loss

def main():
    # initialisation

    jax.distributed.initialize()
    jax_smi.initialise_tracking()
    jax.config.update('jax_platforms', 'cpu')  # suppress TPU in subprocesses
    process_index = jax.process_index()
    if process_index == 0:
        wandb.init(project='bart-pretraining')

    # hyperparameters

    local_devices = jax.local_devices()
    n_local_devices = jax.local_device_count()

    n_epochs = 12

    batch_size_per_device_train = 80
    batch_size_per_device_eval = 80

    eval_every_n_steps = 1024
    save_every_n_steps = 20480

    key = seed2key(seed=42 + process_index)

    sentences_train = load_enwiki(show_progress_bar=process_index == 0)
    sentences_eval = sentences_train[-6400:]  # TODO: split train/eval

    key, subkey = split_key(key)
    preprocessor_train = Preprocessor(sentences_train, key=subkey, batch_size_per_device=batch_size_per_device_train, n_workers=16)

    key, subkey = split_key(key)
    preprocessor_eval = Preprocessor(sentences_eval, key=subkey, batch_size_per_device=batch_size_per_device_eval, n_workers=16)

    key, subkey = split_key(key)
    params = init_params(key=subkey)

    global optimizer
    optimizer = optax.chain(
        optax.adaptive_grad_clip(0.1, eps=0.001),
        optax.sgd(learning_rate=0.03),
    )
    opt_state = optimizer.init(params)

    replicated_params = jax.device_put_replicated(params, local_devices)
    replicated_opt_state = jax.device_put_replicated(opt_state, local_devices)

    for epoch in range(n_epochs):
        if process_index == 0:
            epoch_loss_train = 0.

        for step, batch_train in enumerate(preprocessor_train):
            if process_index == 0:
                start_time = time.time()

            key, subkey = split_key(key); subkeys = split_key(subkey, num=n_local_devices)  # force `subkeys` to be an array instead of a list
            replicated_params, replicated_opt_state, replicated_batch_loss_train = train_step(
                replicated_params,
                replicated_opt_state,
                batch_train.src,
                batch_train.dst,
                batch_train.mask_dec_1d,
                batch_train.mask_enc,
                batch_train.mask_dec,
                batch_train.mask_dec_enc,
                batch_train.labels,
                dropout_key=subkeys,
            )

            if process_index == 0:
                # record loss and time
                batch_loss_train = replicated_batch_loss_train[0].item()
                epoch_loss_train += batch_loss_train
                elapsed_time = time.time() - start_time
                wandb.log({'train loss': batch_loss_train, 'time': elapsed_time}, commit=False)

                # save params
                if step % save_every_n_steps == 0:
                    params = jax.tree_map(lambda x: x[0], replicated_params)
                    filename = f'{wandb.run.name}-{epoch}-{step}.dat'
                    save_params(params, filename)

            # eval
            if step % eval_every_n_steps == 0:
                if process_index == 0:
                    total_loss_eval = 0.

                for batch_eval in preprocessor_eval:
                    replicated_batch_loss_eval = eval_step(
                        replicated_params,
                        batch_eval.src,
                        batch_eval.dst,
                        batch_eval.mask_dec_1d,
                        batch_eval.mask_enc,
                        batch_eval.mask_dec,
                        batch_eval.mask_dec_enc,
                        batch_eval.labels,
                    )
                    if process_index == 0:
                        batch_loss_eval = replicated_batch_loss_eval[0].item()
                        total_loss_eval += batch_loss_eval

                if process_index == 0:
                    wandb.log({'eval loss': total_loss_eval}, commit=False)

            if process_index == 0:
                wandb.log({}, commit=True)

        if process_index == 0:
            epoch_loss_train /= step
            wandb.log({'epoch loss': epoch_loss_train}, commit=False)

            # save params
            params = jax.tree_map(lambda x: x[0], replicated_params)
            filename = f'{wandb.run.name}-{epoch}.dat'
            save_params(params, filename)

if __name__ == '__main__':
    main()
