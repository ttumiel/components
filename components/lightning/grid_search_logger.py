from pytorch_lightning.utilities import rank_zero_only
import numpy as np
import pandas as pd

from components.lightning import TableLogger


class GridSearchLogger(TableLogger):
    def __init__(self, name='GridSearchLogger', version=None, search_params=None, multi_runs=False):
        super().__init__(name, version, multi_runs)
        self.hparams = {}
        self.hparam_version = 0
        self.multi_run_hparams = {}
        self.search_params = search_params
        self.ignore_params = {'train/lr'}

    def after_param_init(self, params):
        if self.search_params is None:
            self.search_params = list(params)
        self.table = pd.DataFrame(columns=['epoch']+self.search_params)

    def average_metrics(self):
        avg_metrics = {key: np.mean([m[key] for m in self.metrics])
                       for key in self.metrics[0].keys()}
        avg_metrics.update(self.val_metrics)
        avg_metrics.update({k:self.hparams[k] for k in self.search_params})
        avg_metrics = {k:v for k,v in avg_metrics.items() if k not in self.ignore_params}

        if self.multi_runs:
            if len(self.multi_params) == avg_metrics['epoch']:
                self.multi_params.append({})
            items = self.multi_params[avg_metrics['epoch']]
            for k,v in avg_metrics.items():
                if k in items: items[k].append(v)
                else:          items[k] = [v]
        else:
            if len(self.table) == self.hparam_version:
                self.table.iloc[self.hparam_version-1] = avg_metrics
            else:
                self.table = self.table.append(avg_metrics, ignore_index=True)

        self.metrics = []

    @rank_zero_only
    def log_hyperparams(self, params):
        if params != self.hparams:
            self.hparam_version += 1
            self.hparams = params.copy()
            self.multi_params = []
            if len(self.table) == 0:
                self.after_param_init(params)

    @rank_zero_only
    def finalize(self, status):
        self.multi_run_hparams[self.hparam_version] = self.multi_params
#         super().finalize(status)
        self.agg_and_log_metrics(None)
        if len(self.metrics) > 0:
            self.average_metrics()

        if self.multi_runs:
            data = [{
                **{k+"/mean":np.mean(q) for k,q in filter(lambda k: k[0] not in self.ignore_params and k[0].startswith(('train','val')), v[-1].items())},
                **{k:np.mean(q) for k,q in filter(lambda k: k[0] in self.ignore_params or not k[0].startswith(('train','val')), v[-1].items())},
                **{k+"/std":np.std(q) for k,q in filter(lambda k: k[0] not in self.ignore_params and k[0].startswith(('train','val')), v[-1].items())}
                } for v in self.multi_run_hparams.values()]
            self.table = pd.DataFrame(data)

        self.display()
        self.save()
