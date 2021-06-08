from collections import Counter
from pathlib import Path
import json 
import random
import torchmetrics                               # For metrics 
from pytorch_lightning.core.lightning import LightningModule  # For model
import torch                                      # For utility
from torch.nn import functional as F              # For loss, activation functions
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from base import BaseDataset, get_dataloaders, get_test_dataloader, BaseNet
from base import BaseNet
from .experiment import ExperimentInterface

# For hyperparameter sweeping individual learners 
class SmartBagging(ExperimentInterface):
  def run_experiment(self, args: dict) -> None:
    # 1) Init Data Components
    train_json = str(Path(args['datastore']) / 'train.json')
    test_json = str(Path(args['datastore']) / 'test.json')
    bags = create_bags(args, train_json)
    dataset = bags[0]  # Hardcoded to first bag, found hp's should be representative
    train_dataloader, val_dataloader = get_dataloaders(dataset, batch_size=args['batch_size'], num_workers=args['num_workers'])
    test_dataloader = get_test_dataloader(args['datastore'], test_json)

    # 2) Init Model
    if args['weighted_loss'] is not None:
      hist = get_histogram(process_json(train_json))
      model = BaseNet(model_type=args['model'], lr=args['lr'], weighted_loss=args['weighted_loss'], hist=hist, beta=args['beta'])
    else:
      model = BaseNet(model_type=args['model'], lr=args['lr'])

    # 3) Init Trainer 
    trainer = Trainer(gpus=args['gpus'], max_epochs=args['epochs'],
                      checkpoint_callback=True, 
                      logger=TensorBoardLogger(save_dir='lightning_logs'))

    # 4) Run Training
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.save_checkpoint("training_end.ckpt")

    # 5) Run Inference
    result = trainer.test(model, test_dataloader)
    print(result)

# Bagging Logic
def process_json(json_path):
  with open(json_path) as f:
    data = json.load(f)  
  return data['annotations']

def get_histogram(input): # list of json data
  if isinstance(input, str):
    json_data = process_json(input)
    category_ids = [entry['category_id'] // 2 for entry in json_data]
    hist = Counter(category_ids)
  else:
    json_data = input
    category_ids = [entry['category_id'] // 2 for entry in json_data]
    hist = Counter(category_ids)
  return hist

def split_histogram(hist, data, threshold):
  # Find class corresponding to threshold
  bimap = dict(reversed(item) for item in hist.items())
  nearest_cnt = min(hist.values(), key=lambda x:abs(x-threshold))
  cl = bimap[nearest_cnt]

  # Divide distribution
  sorted_data = sorted(data, key=lambda x: hist[x['category_id']], reverse=True)  # sort classes by frequency  
  category_ids = [entry['category_id'] // 2 for entry in sorted_data]
  split = category_ids.index(cl) # Find first instance of cl
  majority = sorted_data[:split]
  minority = sorted_data[split:]
  return majority, minority

# This creates a list of BaseDatasets
def create_bags(args, imbalanced_json):
  # 1) Load Data
  entries = process_json(imbalanced_json)

  # 2) Calculate Number of Learners
  hist = get_histogram(entries)
  num_bags = max(hist.values()) // args['threshold']

  # 3) Create Datasets
  majority, minority = split_histogram(hist, entries, args['threshold'])
  random.shuffle(majority) # need shuffle the deck! 
  bags = []
  for start in range(num_bags):
    current_bag = majority[start::num_bags] + minority
    bags.append(BaseDataset())

  return bags