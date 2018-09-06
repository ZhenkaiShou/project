# Simple Implementation of Variational Autoencoder
## Basics
This repository implements a simple variational autoencoder (VAE) to learn the latent representation as well as reconstruct the input on MNIST dataset.
## Performance
We compare the performance between a VAE and an autoencoder (AE) in the following ways:
### 2D Latent Representation of a VAE (left) and an AE (Right):
<p float="center">
  <img src="/stand%20alone%20implementation/VAE/Figures/Latent%20Representation/vae_2.png" alt="VAE" width="45%"/>
  <img src="/stand%20alone%20implementation/VAE/Figures/Latent%20Representation/ae_2.png" alt="AE" width="45%"/>
</p>

### Image Reconstruction of a VAE (left) and an AE (Right) with 16D Latent Representation:
<p float="center">
  <img src="/stand%20alone%20implementation/VAE/Figures/Reconstruction/vae_16.png" alt="VAE" width="45%"/>
  <img src="/stand%20alone%20implementation/VAE/Figures/Reconstruction/ae_16.png" alt="AE" width="45%"/>
</p>

### Linear Interpolation of a VAE in 16D Latent Space:
<p float="center">
  <img src="/stand%20alone%20implementation/RNN/Figures/Mixture Reconstruction/vae_16_2_7.png" alt="VAE" width="75%"/>
</p>

## Reference

- [Intuitively Understanding Variational Autoencoders](https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf)

