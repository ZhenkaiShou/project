# Simple Implementation of Mixture Density Network
## Basics
This repository implements a simple mixture density network (MDN) to predict a 2D distribution. Given an input value *x*, MDN outputs the distribution (probability of a mode *π*, mean *μ*, standard deviation *σ*) of output value *y*.
## Training Dataset
The training dataset is a 2D distribution: x = 7 \* *sin*(0.75 \* *y*) + 0.5 \* *y* + *N*(0, 1)\
![Training Dataset](/stand%20alone%20implementation/MDN/Figures/Training%20and%20Test%20Samples/training_data.png "Training Dataset")
## Performance
### Comparison between the MDN (left) and a fully connected network (FC) (right):
<img src="/stand%20alone%20implementation/MDN/Figures/Training%20and%20Test%20Samples/mdn_5.png" alt="MDN" width="45%"/>
<img src="/stand%20alone%20implementation/MDN/Figures/Training%20and%20Test%20Samples/fc.png" alt="FC" width="45%"/>\
### Influence of the number of modes:
<img src="/stand%20alone%20implementation/MDN/Figures/Training%20and%20Test%20Samples/mdn_1.png" alt="MDN" width="45%"/>
<img src="/stand%20alone%20implementation/MDN/Figures/Training%20and%20Test%20Samples/mdn_3.png" alt="MDN" width="45%"/>\
<img src="/stand%20alone%20implementation/MDN/Figures/Training%20and%20Test%20Samples/mdn_5.png" alt="MDN" width="45%"/>
<img src="/stand%20alone%20implementation/MDN/Figures/Training%20and%20Test%20Samples/mdn_7.png" alt="MDN" width="45%"/>\
### Influence of temperature:
![Temperature](/stand%20alone%20implementation/MDN/Figures/Temperature%20Comparison/mdn_5.gif "Temperature")

## Reference
- [Mixture Density Networks with PyTorch](https://github.com/hardmaru/pytorch_notebooks/blob/master/mixture_density_networks.ipynb)
