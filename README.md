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
- GridAI for rapid experiment training and hyperparameter sweeps.
- learn2learn for meta-learning model support. 