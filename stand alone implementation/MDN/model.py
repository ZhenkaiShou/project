import math
import tensorflow as tf
from config import *

class MDN_Model(object):
  def mdn(self):
    # A mixture density network implementation.
    self.X = tf.placeholder(tf.float32, (None, 1))
    self.Y = tf.placeholder(tf.float32, (None, 1))
    self.LR = tf.placeholder(tf.float32, ())
    
    self.h = tf.layers.dense(self.X, HIDDEN_UNITS, activation = tf.tanh)
    
    # pi(X) = probability of X belonging to a certain mode (s.t. sum up to 1)
    # mu(X) = mean value of the output for that mode
    # sigma(X) = standard deviation of the output for that mode (s.t. sigma > 0)
    self.logits = tf.layers.dense(self.h, MODES)
    self.mu = tf.layers.dense(self.h, MODES)
    self.sigma = tf.layers.dense(self.h, MODES)
    
    self.pi = tf.nn.softmax(self.logits)
    #self.pi = self.pi - tf.reduce_logsumexp(self.logits, 1, keepdims=True)
    self.sigma = tf.exp(self.sigma)
    
    # Minimize the negative log-likelihood of Y for a given X
    # loss = -log[p(Y|X)] 
    #      = -log[sum(p(Y, mode|X))]
    #      = -log[sum(p(mode)p(Y|X, mode))]
    #      = -log[sum(pi(X) * N(Y|mu(X), sigma(X)))]
    # N(Y|mu, sigma) = 1 / (sigma * sqrt(2 * pi)) * exp(-1/2 * ((Y - mu) / sigma)^2)
    # shape(Y) = (None, 1)
    # shape(pi) = (None, MODES)
    # shape(mu) = (None, MODES)
    # shape(sigma) = (None, MODES)
    self.gaussian = 1 / (self.sigma * tf.sqrt(2 * math.pi)) * tf.exp(-0.5 * tf.square((self.Y - self.mu) / self.sigma))
    self.loss = tf.reduce_mean(-tf.log(tf.reduce_sum(self.pi * self.gaussian, -1) + EPSILON))
    
    #logSqrtTwoPI = tf.log(tf.sqrt(2.0 * math.pi))
    #tf_lognormal = -0.5 * ((self.Y - self.mu) / self.sigma) ** 2 - tf.log(self.sigma) - logSqrtTwoPI
    #v = self.pi + tf_lognormal
    #v = tf.reduce_logsumexp(v, 1, keepdims=True)
    #self.loss = -tf.reduce_mean(v)
    
    # Add L2 Regularization loss.
    self.regularization_loss = sum(C_L2 * 2 * tf.nn.l2_loss(var) for var in tf.trainable_variables() if "kernel" in var.name)
    self.total_loss = self.loss + self.regularization_loss
    
    # Optimization.
    optimizer = tf.train.AdamOptimizer(self.LR)
    self.train_op = optimizer.minimize(self.total_loss)
  
  def fc(self):
    # A single-hidden-layer network for comparison.
    self.X = tf.placeholder(tf.float32, (None, 1))
    self.Y = tf.placeholder(tf.float32, (None, 1))
    self.LR = tf.placeholder(tf.float32, ())
    
    self.z = tf.layers.dense(self.X, HIDDEN_UNITS, activation = tf.tanh)
    self.y = tf.layers.dense(self.z, 1)
    
    self.loss = tf.losses.mean_squared_error(self.Y, self.y)
    
    # Add L2 Regularization loss.
    self.regularization_loss = sum(C_L2 * 2 * tf.nn.l2_loss(var) for var in tf.trainable_variables() if "kernel" in var.name)
    tf.losses.add_loss(self.regularization_loss)
    self.total_loss = tf.losses.get_total_loss()
    
    # Optimization.
    optimizer = tf.train.AdamOptimizer(self.LR)
    self.train_op = optimizer.minimize(self.total_loss)