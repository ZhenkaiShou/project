import tensorflow as tf
from config import *

class VAE_Model(object):
  def vae(self):
    # A variational autoencoder implementation.
    self.Inputs = tf.placeholder(tf.float32, (None, INPUT_LENGTH * INPUT_WIDTH))
    
    # Encoding part.
    z0 = tf.layers.dense(self.Inputs, HIDDEN_UNITS[0], activation = tf.nn.relu)
    z1 = tf.layers.dense(z0, HIDDEN_UNITS[1], activation = tf.nn.relu)
    z2 = tf.layers.dense(z1, HIDDEN_UNITS[2], activation = tf.nn.relu)
    
    # mu(X) = mean value of the latent representation z
    # sigma(X) = standard deviation of the latent representation z (sigma > 0)
    self.mu = tf.layers.dense(z2, Z_LENGTH)
    log_sigma = tf.layers.dense(z2, Z_LENGTH)
    self.sigma = tf.exp(log_sigma)
    
    # Sample z from N(mu, sigma^2).
    random = tf.random_normal(tf.shape(self.sigma))
    self.z = self.mu + self.sigma * random
    
    # Decoding part.
    z2_ = tf.layers.dense(self.z, HIDDEN_UNITS[2], activation = tf.nn.relu)
    z1_ = tf.layers.dense(z2_, HIDDEN_UNITS[1], activation = tf.nn.relu)
    z0_ = tf.layers.dense(z1_, HIDDEN_UNITS[0], activation = tf.nn.relu)
    self.outputs_logits = tf.layers.dense(z0_, INPUT_LENGTH * INPUT_WIDTH)
    self.outputs = tf.sigmoid(self.outputs_logits)
    
    # Total loss = reconstruction loss + KL divergence
    # KL(N(mu(X), sigma(X)^2) || N(0, 1)) = 1/2 * [tr(sigma(X)^2) + mu(X)^T * mu(X) - K - log(det(sigma(X)^2))]
    #                                     = 1/2 * sum(sigma_k^2 + mu_i^2 - k - log(sigma_k^2))
    self.reconstruction_loss = INPUT_LENGTH * INPUT_WIDTH * tf.losses.sigmoid_cross_entropy(self.Inputs, self.outputs_logits)
    self.kl_divergence = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(self.sigma) + tf.square(self.mu) - 1 - 2 * log_sigma, -1))
    self.total_loss = self.reconstruction_loss + self.kl_divergence
    
    # Optimization.
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    self.train_op = optimizer.minimize(self.total_loss)
  
  def ae(self):
    # An autoencoder implementation.
    self.Inputs = tf.placeholder(tf.float32, (None, INPUT_LENGTH * INPUT_WIDTH))
    
    # Encoding part.
    z0 = tf.layers.dense(self.Inputs, HIDDEN_UNITS[0], activation = tf.nn.relu)
    z1 = tf.layers.dense(z0, HIDDEN_UNITS[1], activation = tf.nn.relu)
    z2 = tf.layers.dense(z1, HIDDEN_UNITS[2], activation = tf.nn.relu)
    
    # Output latent representation z.
    self.z = tf.layers.dense(z2, Z_LENGTH)
    
    # Decoding part.
    z2_ = tf.layers.dense(self.z, HIDDEN_UNITS[2], activation = tf.nn.relu)
    z1_ = tf.layers.dense(z2_, HIDDEN_UNITS[1], activation = tf.nn.relu)
    z0_ = tf.layers.dense(z1_, HIDDEN_UNITS[0], activation = tf.nn.relu)
    self.outputs_logits = tf.layers.dense(z0_, INPUT_LENGTH * INPUT_WIDTH)
    self.outputs = tf.sigmoid(self.outputs_logits)
    
    # Total loss = reconstruction loss + regression loss
    self.reconstruction_loss = INPUT_LENGTH * INPUT_WIDTH * tf.losses.sigmoid_cross_entropy(self.Inputs, self.outputs_logits)
    self.regularization_loss = sum(C_L2 * 2 * tf.nn.l2_loss(var) for var in tf.trainable_variables() if "kernel" in var.name)
    self.total_loss = self.reconstruction_loss + self.regularization_loss
    
    # Optimization.
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    self.train_op = optimizer.minimize(self.total_loss)
