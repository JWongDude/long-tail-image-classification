from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from base import BaseDataset, get_dataloaders, get_test_dataloader, BaseNet
from base import BaseNet
from .experiment import ExperimentInterface

class Baseline(ExperimentInterface):
  def run_experiment(self, args: dict) -> None:
    # 1) Init Data Components
    train_json = str(Path(args['datastore']) / 'train.json')
    test_json = str(Path(args['datastore']) / 'test.json')
    dataset = BaseDataset(args['datastore'], train_json, image_size=args['image_size'], da=args['augment_data'])

    train_dataloader, val_dataloader = get_dataloaders(dataset, batch_size=args['batch_size'], num_workers=args['num_workers'])
    test_dataloader = get_test_dataloader(args['datastore'], test_json)

    # 2) Init Model
    model = BaseNet(model_type=args['model'], lr=args['lr'], weight_class=args['weight_class'], weight=args['weight'])

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