from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from base import BaseDataset, get_dataloaders, get_test_dataloader, BaseNet
from base import BaseNet
from .experiment import ExperimentInterface

class DoubleModel(ExperimentInterface):
  def run_experiment(self, args: dict) -> None:
    pass