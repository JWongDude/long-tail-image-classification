# Long-Tailed Image Classification

## Problem Statement:
Real-world training datasets have unequal long-tailed distributions,
in which training samples from some categories are more prevalent than others.

Provided a skewed training distribution, can you train a model with high 
classification accuracy?

## Setup:
CIFAR-50 long-tailed distribution is produced from CIFAR-100 dataset
with varying decay rates: [0.995, 0.98, 0.9, 0]

## Tools:
- Pytorch Lightning for automated training loops and ML boilerplate.
- Timm Models for SOTA pretrained image classification backbones. 
- GridAI for experiment tracking and hyperparameter sweeps on the cloud.

## Results:
(Add photo here)

## Project Documentation: 
For futher details, please access the final report and presentation located inside documentation folder.
To see experiment development, please access the "Data Preparation" and "Modeling" notebooks inside the notebook folder.
To see core code, please access the "base" folder for fundamental backbone of all experimentation. 