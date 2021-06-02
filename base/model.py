import torch                                      # For utility
from torch import nn                              # For layers
from torch.nn import functional as F              # For loss, activation functions
import timm                                       # For transfer learning
import torchmetrics                               # For metrics 
from pytorch_lightning.core.lightning import LightningModule  # For model

class BaseNet(LightningModule):
  def __init__(self, model_type='efficientnet_b0', lr=1e-5):
    super().__init__()
    self.save_hyperparameters()

    # Create backbone and classifier
    backbone = timm.create_model(model_name=model_type, pretrained=True)
    layers = list(backbone.children())[:-1]
    [fc] = list(backbone.children())[-1:]
    self.feature_extractor = nn.Sequential(*layers)
    self.classifer = nn.Linear(fc.in_features, 50)

    # Define metrics
    self.train_acc = torchmetrics.Accuracy()
    self.valid_acc = torchmetrics.Accuracy()
    self.test_acc = torchmetrics.Accuracy()

  def forward(self, x):
    self.feature_extractor.eval() 
    with torch.no_grad():
      representations = self.feature_extractor(x).flatten(1)

    # Learn Head
    x = self.classifer(representations) 
    return x  

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), self.hparams['lr'])

  def training_step(self, batch, batch_idx):
    x, y = batch 
    logits = self(x)
    loss = F.cross_entropy(logits, y) # One hot encoding, log_softmax interally
    preds = F.softmax(logits, dim=1)
    self.train_acc(preds, y)

    self.log('train_loss', loss)
    self.log('train_acc', self.train_acc)
    return loss  # Mandatory

  def validation_step(self, batch, batch_idx):
    x, y = batch 
    logits = self(x)
    loss = F.cross_entropy(logits, y)
    preds = F.softmax(logits, dim=1)
    self.valid_acc(preds, y)

    self.log('val_loss', loss)
    self.log('val_acc', self.valid_acc)

  def test_step(self, batch, batch_idx):
    x, y = batch 
    logits = self(x)
    loss = F.cross_entropy(logits, y)
    preds = F.softmax(logits, dim=1)
    self.test_acc(preds, y)

    self.log('test_loss', loss)
    self.log('test_acc', self.test_acc)