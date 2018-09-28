import math
import numpy as np
import tensorflow as tf

from config import *

class SteinVAE_Model(object):
  def steinvae(self):
    # A variational autoencoder implementation.
    self.Inputs = tf.placeholder(tf.float32, (None, INPUT_LENGTH * INPUT_WIDTH))
    
    # Encoding part.
    z0 = tf.layers.dense(self.Inputs, HIDDEN_UNITS[0], activation = tf.nn.relu)
    z1 = tf.layers.dense(z0, HIDDEN_UNITS[1], activation = tf.nn.relu)
    z2 = tf.layers.dense(z1, HIDDEN_UNITS[2], activation = tf.nn.relu)
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
    
    # Compute the pairwise distance of x.
    r = tf.reduce_sum(self.z * self.z, 1, keepdims = True)
    distance_sqr = tf.stop_gradient(r) - 2 * tf.matmul(tf.stop_gradient(self.z), tf.transpose(self.z)) + tf.transpose(r)
    ones = tf.ones_like(distance_sqr)
    mask_upper = tf.matrix_band_part(ones, 0, -1)
    mask_diagonal = tf.matrix_band_part(ones, 0, 0)
    mask = tf.cast(mask_upper - mask_diagonal, dtype = tf.bool)
    pdist_sqr = tf.boolean_mask(distance_sqr, mask)
    
    # Compute the median of the pairwise distance.
    m = tf.shape(pdist_sqr)[0] // 2
    values, _ = tf.nn.top_k(pdist_sqr, tf.shape(pdist_sqr)[0])
    med_sqr = values[m]
    
    # Compute the kernel.
    h = med_sqr / tf.log(tf.cast(tf.shape(self.Inputs)[0], tf.float32))
    # shape(kernel) = (batch_size, batch_size)
    kernel = tf.exp(-1.0 / tf.stop_gradient(h) * distance_sqr)
    # shape(grad_kernel[i]) = (batch_size, Z_LENGTH)
    grad_kernel = [tf.gradients(kernel[i], self.z)[0] for i in range(BATCH_SIZE)]
    
    # Compute the probability.
    mu = 0.0
    sigma = 1.0
    gaussian = 1.0 / (sigma * tf.sqrt(2 * math.pi)) * tf.exp(-0.5 * tf.square(((self.z - mu) / sigma)))
    grad_log_p = tf.gradients(tf.reduce_sum(tf.log(gaussian + EPSILON), 1), self.z)[0]
    
    # Gradients for SVGD.
    grad_svgd = tf.stack([-(tf.reduce_mean(tf.reshape(kernel[i], [-1, 1]) * grad_log_p, 0) + tf.reduce_mean(grad_kernel[i], 0)) for i in range(BATCH_SIZE)])
    vars = tf.trainable_variables()
    grad_encoding = tf.gradients(self.z, vars, grad_ys = grad_svgd)
    
    # Reconstruction loss.
    self.reconstruction_loss = INPUT_LENGTH * INPUT_WIDTH * tf.losses.sigmoid_cross_entropy(self.Inputs, self.outputs_logits)
    
    # KL Divergence (for evaluation purpose).
    self.kl_divergence = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(self.sigma) + tf.square(self.mu) - 1 - 2 * log_sigma, -1))

    # Optimize the loss.
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    train_op1 = optimizer.apply_gradients(zip(grad_encoding, vars))
    train_op2 = optimizer.minimize(self.reconstruction_loss)
    self.train_op = tf.group(train_op1, train_op2)    
