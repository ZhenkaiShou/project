import tensorflow as tf

import config

# Local hyperparameters
Z_LENGTH = config.Z_LENGTH
KL_TOLERANCE = 0.0

class VAE(object):
  def __init__(self, name = "vae"):
    self.name = name
    self.build_model()
  
  def build_model(self):
    with tf.variable_scope(self.name):
      self.Input = tf.placeholder(tf.float32, (None, 64, 64, 3))
      
      # Encoding.
      e_conv1 = tf.layers.conv2d(self.Input, 32, (4, 4), (2, 2), activation = tf.nn.relu, name = "e_conv1")
      e_conv2 = tf.layers.conv2d(e_conv1, 64, (4, 4), (2, 2), activation = tf.nn.relu, name = "e_conv2")
      e_conv3 = tf.layers.conv2d(e_conv2, 128, (4, 4), (2, 2), activation = tf.nn.relu, name = "e_conv3")
      e_conv4 = tf.layers.conv2d(e_conv3, 256, (4, 4), (2, 2), activation = tf.nn.relu, name = "e_conv4")
      e_conv4 = tf.layers.flatten(e_conv4)
      
      # Mean and standard deviation.
      self.mu = tf.layers.dense(e_conv4, Z_LENGTH, name = "mu")
      log_sigma = tf.layers.dense(e_conv4, Z_LENGTH, name = "log_sigma")
      self.sigma = tf.exp(log_sigma)
      
      # Sample z from N(mu, sigma^2).
      random = tf.random_normal(tf.shape(self.sigma))
      self.z = self.mu + self.sigma * random
      
      # Decoding.
      d_fc = tf.layers.dense(self.z, 1024, name = "d_fc")
      d_fc = tf.reshape(d_fc, (-1, 1, 1, 1024))
      d_conv3 = tf.layers.conv2d_transpose(d_fc, 128, (5, 5), (2, 2), activation = tf.nn.relu, name = "d_conv3")
      d_conv2 = tf.layers.conv2d_transpose(d_conv3, 64, (5, 5), (2, 2), activation = tf.nn.relu, name = "d_conv2")
      d_conv1 = tf.layers.conv2d_transpose(d_conv2, 32, (6, 6), (2, 2), activation = tf.nn.relu, name = "d_conv1")
      output_logits = tf.layers.conv2d_transpose(d_conv1, 3, (6, 6), (2, 2), name = "output")
      self.output = tf.nn.sigmoid(output_logits)
    self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.name)
  
    # Loss function.
    self.reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.output - self.Input), [1, 2, 3]))
    kl_divergence = 0.5 * tf.reduce_sum(tf.square(self.sigma) + tf.square(self.mu) - 1 - 2 * log_sigma, -1)
    self.kl_divergence = tf.reduce_mean(kl_divergence)
    kl_divergence = tf.reduce_mean(tf.maximum(kl_divergence, KL_TOLERANCE * Z_LENGTH))
    self.loss = self.reconstruction_loss + kl_divergence
  
  def build_training(self):
    # Optimization.
    self.LR = tf.placeholder(tf.float32, ())
    optimizer = tf.train.AdamOptimizer(self.LR)
    self.train_op = optimizer.minimize(self.loss, var_list = self.trainable_variables)