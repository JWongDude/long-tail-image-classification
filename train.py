from argparse import ArgumentParser
import pytorch_lightning.utilities.seed as seed
from experiment import Baseline, SmartBagging

# Experiments 
def run_experiment(args: dict) -> None:
  experiments = {'baseline': Baseline(), 
                'smart_bagging': SmartBagging()}

  experiments[args['experiment']].run_experiment(args)

def main():
  seed.seed_everything(1)

  # Specify Experiment
  parser = ArgumentParser()
  parser.add_argument('--experiment', type=str, default='baseline') # Other option is smart_bagging

  # Data args 
  parser.add_argument('--datastore', type=str)  # COPY-PASTE EXACT DATASTORE NAME HERE
  parser.add_argument('--augment_data', type=bool, default=False)
  parser.add_argument('--image_size', type=int, default=224)
  parser.add_argument('--batch_size', type=int, default=16)
  parser.add_argument('--num_workers', type=int, default=4)
  parser.add_argument('--threshold', type=int, default=0) # Configure this for smart bagging 

  # Model args
  parser.add_argument('--model', type=str, default='efficientnet_b0')
  parser.add_argument('--lr', type=float, default=4e-4)
  parser.add_argument('--weighted_loss', type=str, default=None)
  parser.add_argument('--beta', type=float, default=None)

  # Trainer args 
  parser.add_argument('--gpus', type=int, default=1)
  parser.add_argument('--epochs', type=int, default=5)

  # Run experiment
  args = parser.parse_args()
  run_experiment(dict(vars(args)))

if __name__ == '__main__': 
  main()