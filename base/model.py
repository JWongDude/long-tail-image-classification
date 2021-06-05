from pytorch_lightning.core.step_result import weighted_mean
import torch                                      # For utility
from torch.nn import functional as F              # For loss, activation functions
from torch import nn                              # For layers
import timm                                       # For transfer learning
import torchmetrics                               # For metrics 
from pytorch_lightning.core.lightning import LightningModule  # For model

class BaseNet(LightningModule):
  def __init__(self, model_type='efficientnet_b0', lr=4e-4, weighted_loss=None, hist=None, beta=None):
    super().__init__()
    self.save_hyperparameters()

    # Create backbone
    backbone = timm.create_model(model_type, pretrained=True)
    layers = list(backbone.children())[:-1]
    [fc] = list(backbone.children())[-1:]
    self.feature_extractor = nn.Sequential(*layers)
    self.classifer = nn.Linear(fc.in_features, 50)

    # Weighted Loss
    self.weighted_loss = WeightedLoss(weighted_loss, hist, beta)

    # Metrics
    self.train_acc = torchmetrics.Accuracy()
    self.valid_acc = torchmetrics.Accuracy()
    self.test_acc = torchmetrics.Accuracy() 
    self.preds = []
    self.labels = []

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
    # loss = F.cross_entropy(logits, y)  # One hot encoding, log_softmax interally
    loss = self.weighted_loss(F.log_softmax(logits), y)  # Identity or Weighted Function
    preds = F.softmax(logits, dim=1)
    self.train_acc(preds, y)

    self.log('train_loss', loss)
    self.log('train_acc', self.train_acc)
    return loss  # Goes to optimizer

  def validation_step(self, batch, batch_idx):
    x, y = batch 
    logits = self(x)
    # loss = F.cross_entropy(logits, y)
    loss = self.weighted_loss(F.log_softmax(logits), y)
    preds = F.softmax(logits, dim=1)
    self.valid_acc(preds, y)

    self.log('val_loss', loss)
    self.log('val_acc', self.valid_acc)

  def test_step(self, batch, batch_idx):
    x, y = batch 
    logits = self(x)
    # loss = F.cross_entropy(logits, y)
    loss = self.weighted_loss(F.log_softmax(logits), y)
    preds = F.softmax(logits, dim=1)
    self.test_acc(preds, y)

    self.log('test_loss', loss)
    self.log('test_acc', self.test_acc)

    # Aggregated Metrics
    self.preds.append(preds.cpu())
    self.labels.append(y.cpu())

  def test_epoch_end(self, outputs):
    precision = torchmetrics.Precision(num_classes=50)
    precision(torch.cat(self.preds), torch.cat(self.labels))
    self.log('precision', precision)

    recall = torchmetrics.Recall(num_classes=50)
    recall(torch.cat(self.preds), torch.cat(self.labels))
    self.log('recall', recall)

    auroc = torchmetrics.AUROC(num_classes=50)
    auroc(torch.cat(self.preds), torch.cat(self.labels))
    self.log('AUROC', auroc)
  
# Weighted Loss Functions
class WeightedLoss:
  def __init__(self, weighted_loss, hist, beta=None):
    # Store arguments, Calculate Normalized Weights
    if weighted_loss == "ins":
      weights = torch.tensor([1.0 / sample_count for sample_count in hist.values()])
      self.weight_map = weights / torch.sum(weights)  

    elif weighted_loss == "isns":
      weights = torch.sqrt(torch.tensor([1.0 / sample_count for sample_count in hist.values()]))
      self.weight_map = weights / torch.sum(weights)

    elif weighted_loss == "ens":
      e_numerator = 1.0 - torch.tensor([torch.pow(beta, sample_count) for sample_count in hist.values()])
      e_denominator = 1.0 - beta
      weights = e_denominator / e_numerator

      self.weight_map = weights / torch.sum(weights)
    
    else:  # Identity
      self.weight_map = torch.ones(50)

  # Note, weights are applied to elements of log-softmax, then pass through cross-entopy loss.
  def __call__(self, log_softmax_logits, targets):
    weights = torch.tensor([self.weight_map[target.item()] for target in targets]).cuda()
    return F.nll_loss(log_softmax_logits * weights)