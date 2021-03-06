# Heating up Decision Boundaries

This repository contains the source code of the ICLR 2021 paper 'Heating up decision boundaries: some aspects of adversarial examples, compression and generalization bounds. Please find the full paper at https://openreview.net/forum?id=UwGY2qjqoLD.

In this work, we consider an MLP model as proof of principle trained on planar geometric data, LeNet-5 and basic CNN architectures trained on MNIST and finally a Wide Residual Network 28-10 and 32, 44 and 56 layered ResNets trained on CIFAR10. Please find the respective code for all training methods in directories 'models' and 'train'. 

Our analysis of said models includes mainly an evaluation of the isocapacitory and isoperimetric bound. For the Lenet-5 model we also considered the models robustness for various kind of data corruption. Both parts of the analysis can be found in the 'explorations' directory.

### Requirements

- **torch 1.5.0**
- **torchvision 0.6.0**
