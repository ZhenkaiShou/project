import tensorflow as tf

from config import *

class Actor(object):
  def __init__(self, name):
    self.name = name
    self.build_model()
  
  def build_model(self):
    with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
      self.S = tf.placeholder(tf.float32, (None, S_LENGTH))
      
      # Actor part: the policy network.
      z = self.S
      for _ in range(HIDDEN_LAYER):
        z = tf.layers.dense(z, Z_LENGTH, activation = tf.nn.relu)
      self.pi = ACTION_SCALING * tf.layers.dense(z, A_LENGTH, activation = tf.tanh, kernel_initializer = tf.random_uniform_initializer(minval = -1e-3, maxval = 1e-3))
    self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.name)
    
  def build_training(self, actor_loss):
    # actor loss = -Q(s, pi(s))
    self.actor_loss = actor_loss
    # Optimization.
    self.LR = tf.placeholder(tf.float32, ())
    optimizer = tf.train.AdamOptimizer(self.LR)
    self.train_op = optimizer.minimize(self.actor_loss, var_list = self.trainable_variables)

class Critic(object):
  def __init__(self, name, A = None):
    self.name = name
    self.build_model(A)
  
  def build_model(self, A):
    with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
      self.S = tf.placeholder(tf.float32, (None, S_LENGTH))
      if A is None:
        self.A = tf.placeholder(tf.float32, (None, A_LENGTH))
      else:
        self.A = A
      
      # Critic part: the value network.
      z = tf.concat([self.S, self.A], 1)
      for _ in range(HIDDEN_LAYER):
        z = tf.layers.dense(z, Z_LENGTH, activation = tf.nn.relu)
      self.q = tf.layers.dense(z, 1, kernel_initializer = tf.random_uniform_initializer(minval = -1e-3, maxval = 1e-3))
    self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.name)
    
    # Compute the critic loss.
    self.QTarget = tf.placeholder(tf.float32, (None, 1))
    self.critic_loss = tf.reduce_mean(tf.square(self.QTarget - self.q))
    
    # Compute optional actor loss.
    self.actor_loss = -tf.reduce_mean(self.q)
    
  def build_training(self):
    # Optimization.
    self.LR = tf.placeholder(tf.float32, ())
    with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
      optimizer = tf.train.AdamOptimizer(self.LR)
      self.train_op = optimizer.minimize(self.critic_loss, var_list = self.trainable_variables)