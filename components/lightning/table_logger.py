from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import LightningLoggerBase
import pandas as pd
from IPython.display import DisplayHandle

class TableLogger(LightningLoggerBase):
    def __init__(self, name='TableLogger', version=None):
        super().__init__()
        self.table = pd.DataFrame()
        self.metrics = []
        self.val_metrics = {}
        self.display_handle = DisplayHandle()
        self._version = version
        self._experiment = self.table
        self._name = name

    def average_metrics(self):
        avg_metrics = {key: np.mean([m[key] for m in self.metrics])
                       for key in self.metrics[0].keys()}
        avg_metrics.update(self.val_metrics)
        self.table = self.table.append(avg_metrics, ignore_index=True)
        self.metrics = []

    def display(self):
        if len(self.table) == 1:
            self.display_handle.display(self.table)
        else:
            self.display_handle.update(self.table)

    @rank_zero_only
    def log_hyperparams(self, params):
        # Save hparams into the logger dir
        # Calls logger.save afterwards
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        if 'val/loss' in metrics:
            self.val_metrics = metrics
            self.average_metrics()
            self.display()
        else:
            self.metrics.append(metrics)

    @rank_zero_only
    def finalize(self, status):
        self.agg_and_log_metrics(None)
        if len(self.metrics) > 0:
            self.average_metrics()
            self.display_handle.update(self.table)
        self.save()

    def save(self):
        self.table.to_csv(f'{self.name}/version_{self.version}/logs.csv', index=False)

    @property
    def experiment(self):
        return self._experiment

    @property
    def name(self):
        return self._name

    @property
    def version(self):
        if self._version is None:
            self._version = max(int(re.search(r'\d+$', str(v)).group()) for v in Path(self.name).iterdir())+1
        return self._version
