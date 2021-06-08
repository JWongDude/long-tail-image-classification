# Long-Tailed Image Classification

## Problem Statement:
Real-world training datasets have unequal long-tailed distributions,
in which training samples from some categories are more prevalent than others.

Provided a skewed training distribution, can you train a model with high 
classification accuracy?

Training Datasets:
![image](https://user-images.githubusercontent.com/54962990/121232352-5b4d7180-c846-11eb-92e2-cda8b0156000.png)

Test Dataset:
![image](https://user-images.githubusercontent.com/54962990/121232556-96e83b80-c846-11eb-8353-b48ddd100c11.png)

## Setup:
CIFAR-50 long-tailed distribution is produced from CIFAR-100 dataset
with varying decay rates: [0.995, 0.98, 0.9, 0]

## Tools:
- Pytorch Lightning for automated training loops and ML boilerplate.
- Timm Models for SOTA pretrained image classification backbones. 
- GridAI for experiment tracking and hyperparameter sweeps on the cloud.

## Results:
Through my experimentation, I found stacking using a weighted loss function achieved the highest results of ~8% accuracy improvement accross all datasets. 
![image](https://user-images.githubusercontent.com/54962990/121232918-eb8bb680-c846-11eb-9689-959c6c2aa5a8.png)

Stacking with weighted loss produces the baseline accuracy of the next imbalance class up.
![image](https://user-images.githubusercontent.com/54962990/121233204-40c7c800-c847-11eb-845d-dc53de99c70e.png)

## Project Documentation: 
For futher details, please access the final report and presentation located inside documentation folder.
To see experiment development, please access the "Data Preparation" and "Modeling" notebooks inside the notebook folder.
To see core code, please access the "base" folder for fundamental backbone of all experimentation. 
