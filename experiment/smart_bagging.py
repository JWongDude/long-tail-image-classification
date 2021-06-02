from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from base import BaseDataset, get_dataloaders, get_test_dataloader, BaseNet
from base import BaseNet
from .experiment import ExperimentInterface

class SmartBagging(ExperimentInterface):
  def run_experiment(self, args: dict) -> None:
    # Split into deterministic datasets and train the specified bag num


    pass

# Bagging Logic

