import torch
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from components.lightning import TableLogger, GridSearchLogger, Telemetry
from components.search import grid_search, is_search

# TODO: add telemetry hook
# add nrepeats so that you can do an experiment n times,returning the std, mean
def train(module, *args, max_epochs=2, logger='table_logger',
          accumulate_grad_batches=1, train_percent_check=1.0, weights_summary='top',
          gpus=torch.cuda.device_count(), track_grad_norm=False, save_top_k=3,
          telemetry=False, callbacks=None, **kwargs):

    if is_search(module.hparams) and logger == 'table_logger':
        logger = GridSearchLogger()
        weights_summary = None
    elif logger == 'table_logger':
        logger = TableLogger()
    elif logger == 'tensorboard_logger':
        logger = ...

    if callbacks is None:
        callbacks = []

    if telemetry:
        # Hook into all conv layers and add layer telemetry for the activations and gradients.
        # could also add tensorboard histogram of layers
        t = Telemetry(module, torch.nn.Conv2d)

    # callbacks += [LearningRateLogger()]

    if is_search(module.hparams):
        checkpoint_callback = ModelCheckpoint(
            filepath=logger.path,
            save_top_k=save_top_k,
            verbose=True,
            monitor='val_loss',
            mode='min',
            prefix=''
        )
        logger.search_params = [k for k,v in module.hparams.items() if isinstance(v, list)]

        for hparams in grid_search(module.hparams):
            module.hparams = hparams

            trainer = pl.Trainer(
            max_epochs=max_epochs, logger=logger,
            accumulate_grad_batches=accumulate_grad_batches,
            train_percent_check=train_percent_check, gpus=gpus,
            weights_summary=weights_summary, track_grad_norm=track_grad_norm,
            checkpoint_callback=checkpoint_callback,
            callbacks=callbacks,
            **kwargs
            )
            trainer.fit(module)
            if telemetry:
                t.plot()
                t.reset()
    else:
        checkpoint_callback = ModelCheckpoint(
            filepath=logger.path,
            save_top_k=save_top_k,
            verbose=True,
            monitor='val_loss',
            mode='min',
            prefix=''
        )

        trainer = pl.Trainer(
            max_epochs=max_epochs, logger=logger,
            accumulate_grad_batches=accumulate_grad_batches,
            train_percent_check=train_percent_check, gpus=gpus,
            weights_summary=weights_summary, track_grad_norm=track_grad_norm,
            checkpoint_callback=checkpoint_callback,
            callbacks=callbacks,
            **kwargs
        )
        trainer.fit(module)
        if telemetry:
            t.plot()

    if telemetry:
        t.remove()
