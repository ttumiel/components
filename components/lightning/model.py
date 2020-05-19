import torch
import pytorch_lightning as pl
import multiprocessing as mp
from torch.utils.data import DataLoader
from components.utils import AttrDict


default_hparams = AttrDict({
    'lr': 1e-3,
    'bs': 64,
    'wd': 1e-6,
    'num_workers': mp.cpu_count()
})

class LightningModel(pl.LightningModule):
    """
    A lightning module that can be used to easily set up and run
    reproducible experiments. Just pass in the model and the
    datasets and you're good to go.

    You can also override the `prepare_data` method and set up
    the datasets there:
        self.train_ds = ...
        self.val_ds = ...
        self.test_ds = ...
    """
    def __init__(self, hparams, model=None, loss_fn=None, metrics=[], train_ds=None,
                 val_ds=None, test_ds=None, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds

        p = {**default_hparams}
        p.update(hparams)
        self.hparams = p

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)

        metrics = {f'train/{f.__name__}': f(preds, y) for f in self.metrics}
        logs = OrderedDict({
            'train/loss': loss,
            'train/lr': self.trainer.optimizers[0].param_groups[0]['lr'],
            **metrics
        })
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)

        metrics = {f'val/{f.__name__}': f(preds, y) for f in self.metrics}
        logs = OrderedDict({
            'val/loss': loss,
            **metrics
        })
        return {'val_loss': loss, 'log': logs}

    def train_epoch_end(self,):
        # TODO: save results, rsync to safety?
        pass

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([o['val_loss'] for o in outputs]).mean()
        logs = {key:torch.stack([logs['log'][key] for logs in outputs]).mean()
                for key in outputs[0]['log'].keys()}

        return {'val_loss': avg_loss, 'log': logs}

    def configure_optimizers(self):
        """
        Optimizers can be adjusted and combined by providing a list.

        The schedulers can run at different rates, specified in a dict:
        {
            'scheduler': lr_scheduler,
            'interval': 'step'  # or 'epoch'
            'monitor': 'val_f1',
            'frequency': x
        }
        """
        optim = torch.optim.Adam(self.parameters(), lr=self.hparams['lr'], weight_decay=self.hparams['wd'])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=2, cooldown=1, min_lr=5e-5)
        return [optim], [sched]

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.hparams['bs'],
                          num_workers=self.hparams['num_workers'],
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          batch_size=self.hparams['bs']*2,
                          num_workers=self.hparams['num_workers'])

    def test_dataloader(self):
        return DataLoader(self.test_ds,
                          batch_size=self.hparams['bs']*2,
                          num_workers=self.hparams['num_workers'])

    def test_step(self, batch, batch_idx):
        if isinstance(batch, tuple):
            batch,y = batch
        preds = self(batch)
        return {'preds': preds}

    def test_epoch_end(self, outputs):
        preds = torch.cat([x['preds'] for x in outputs])
        return {'preds': preds}
