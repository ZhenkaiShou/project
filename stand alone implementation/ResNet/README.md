# Simple Implementation of Residual Networks
## Basics
This repository implements simple Residual Networks (ResNet) to classify CIFAR-10 dataset. More details can be found in [this post](https://zhenkaishou.github.io/my-site/deep%20learning/2019/01/28/Deep-Learning-Experiments-on-CIFAR-10-Dataset/).

## Usage
Run `training.py` to train the model. The CIFAR-10 dataset will be automatically downloaded if it does not exist.
```
python training.py
```

To change hyperparameters, please pass all necessary hyperparameters to the arguments of `train()` method in `training.py`. For example, to enable data augmentation:
```
train(file_name = "res4, augment data", flip_data = True, crop_data = True)
```

## Performance
Without tuning the hyperparameters, this simple network (in total 10 layers) can achieve around 13% test error rate.
<img src="/stand%20alone%20implementation/ResNet/Figures/Res4.png" alt="res4" width="100%"/>

## Reference
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)


