# Simple Implementation of Stein Learning Variational Autoencoder
## Basics
This repository implements a simple Stein Learning variational autoencoder (SteinVAE) to learn the latent representation as well as reconstruct the input on MNIST dataset.
## Performance
We compare the performance between a VAE and a SteinVAE in the following ways:

### Training Loss of a VAE (left) and an SteinVAE (Right) with 16D Latent Representation:
SteinVAE has higher KL divergence but lower reconstruction loss. 
<p float="center">
  <img src="/stand%20alone%20implementation/SteinVAE/Figures/Average%20Loss/vae_16.png" alt="VAE" width="50%"/>
  <img src="/stand%20alone%20implementation/SteinVAE/Figures/Average%20Loss/steinvae_16.png" alt="SteinVAE" width="50%"/>
</p>

### 2D Latent Representation of a VAE (left) and a SteinVAE (Right):
The latent representations of both VAE and SteinVAE are centered at the origin.
<p float="center">
  <img src="/stand%20alone%20implementation/SteinVAE/Figures/Latent%20Representation/vae_2.png" alt="VAE" width="45%"/>
  <img src="/stand%20alone%20implementation/SteinVAE/Figures/Latent%20Representation/steinvae_2.png" alt="SteinAE" width="45%"/>
</p>

### Image Reconstruction of a VAE (left) and an SteinVAE (Right) with 16D Latent Representation:
Reconstructions from VAE are more blurry than the reconstructions from SteinVAE.
<p float="center">
  <img src="/stand%20alone%20implementation/SteinVAE/Figures/Reconstruction/vae_16.gif" alt="VAE" width="45%"/>
  <img src="/stand%20alone%20implementation/SteinVAE/Figures/Reconstruction/steinvae_16.gif" alt="SteinAE" width="45%"/>
</p>

## Reference
- [Stein Variational Gradient Descent (the basics)](https://arxiv.org/abs/1608.04471)
- [Amortized_SVGD](https://github.com/lewisKit/Amortized_SVGD/blob/master/SteinVAE/steinvae.py)


