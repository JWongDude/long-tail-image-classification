from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import numpy as np
from .dataset import BaseDataset

def get_dataloaders(dataset, batch_size=16, num_workers=2):
  # Calculate Split
  val_split = 0.2   # Hardcoded 20%
  dataset_size = len(dataset)
  indices = list(range(dataset_size))
  split = int(np.floor(val_split * dataset_size)) 

  # Shuffle Data
  np.random.shuffle(indices)

  # Split Base Dataset into Training and Validation Datasets
  train_indices, val_indices = indices[split:], indices[:split]
  train_sampler = SubsetRandomSampler(train_indices)
  val_sampler = SubsetRandomSampler(val_indices)

  # Dataloaders
  train_dataloader = DataLoader(dataset, batch_size, num_workers=num_workers, sampler=train_sampler)
  val_dataloader = DataLoader(dataset, batch_size, num_workers=num_workers, sampler=val_sampler)

  return train_dataloader, val_dataloader

def get_test_dataloader(datastore, test_json):
  testset = BaseDataset(datastore, test_json)
  test_dataloader = DataLoader(testset, batch_size=32, num_workers=2)
  return test_dataloader