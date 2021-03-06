# Simple Implementation of Mixture Density Network
## Basics
This repository implements a simple mixture density network (MDN) to predict a 2D distribution. Given an input value *x*, MDN returns the distribution (probability of a mode *π*, mean *μ*, standard deviation *σ*) of the output value *y*:<br />
<p align="center">
  <i>π, μ, σ = f(x)</i>
</p>

The output value *y* is then sampled from the distribution (*π, μ, σ*) with temperature *τ*.
## Training Dataset
The training dataset is a 2D distribution: x = 7 \* *sin*(0.75 \* *y*) + 0.5 \* *y* + *N*(0, 1)<br />
![Training Dataset](/stand%20alone%20implementation/MDN/Figures/Training%20and%20Test%20Samples/training_data.png "Training Dataset")
## Performance
### Comparison between the MDN (left) and a fully connected network (FC) (right):
<p float="center">
  <img src="/stand%20alone%20implementation/MDN/Figures/Training%20and%20Test%20Samples/mdn_5.png" alt="MDN" width="45%"/>
  <img src="/stand%20alone%20implementation/MDN/Figures/Training%20and%20Test%20Samples/fc.png" alt="FC" width="45%"/>
</p>

### Influence of the number of modes:
<p float="center">
  <img src="/stand%20alone%20implementation/MDN/Figures/Training%20and%20Test%20Samples/mdn_1.png" alt="MDN" width="45%"/>
  <img src="/stand%20alone%20implementation/MDN/Figures/Training%20and%20Test%20Samples/mdn_3.png" alt="MDN" width="45%"/><br />
  <img src="/stand%20alone%20implementation/MDN/Figures/Training%20and%20Test%20Samples/mdn_5.png" alt="MDN" width="45%"/>
  <img src="/stand%20alone%20implementation/MDN/Figures/Training%20and%20Test%20Samples/mdn_7.png" alt="MDN" width="45%"/>
</p>

### Influence of temperature:
![Temperature](/stand%20alone%20implementation/MDN/Figures/Temperature%20Comparison/mdn_5.gif "Temperature")

## Reference
- [Mixture Density Networks with PyTorch](https://github.com/hardmaru/pytorch_notebooks/blob/master/mixture_density_networks.ipynb)
