import tensorflow as tf

import config

# Local hyperparameters
Z_LENGTH = config.Z_LENGTH
A_LENGTH = config.A_LENGTH
ACTION_SCALING = config.ACTION_SCALING
H_LENGTH = 200
HIDDEN_LAYER = 4

class Actor(object):
  def __init__(self, name):
    self.name = name
    self.build_model()
  
  def build_model(self):
    with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
      self.Z = tf.placeholder(tf.float32, (None, Z_LENGTH))
      self.ZGoal = tf.placeholder(tf.float32, (None, Z_LENGTH))
      
      # Actor part: the policy network.
      h = tf.concat([self.Z, self.ZGoal], 1)
      for _ in range(HIDDEN_LAYER):
        h = tf.layers.dense(h, H_LENGTH, activation = tf.nn.relu)
      self.pi = ACTION_SCALING * tf.layers.dense(h, A_LENGTH, use_bias= False, activation = tf.tanh, kernel_initializer = tf.random_uniform_initializer(minval = -1e-3, maxval = 1e-3))
    self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.name)
  
  def build_training(self, actor_loss):
    # actor loss = -Q(z, pi(z, z_g), z_g)
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
      self.Z = tf.placeholder(tf.float32, (None, Z_LENGTH))
      if A is None:
        self.A = tf.placeholder(tf.float32, (None, A_LENGTH))
      else:
        self.A = A
      self.ZGoal = tf.placeholder(tf.float32, (None, Z_LENGTH))
      
      # Critic part: the value network.
      h = tf.concat([self.Z, self.A, self.ZGoal], 1)
      for _ in range(HIDDEN_LAYER):
        h = tf.layers.dense(h, H_LENGTH, activation = tf.nn.relu)
      self.q = tf.layers.dense(h, 1, kernel_initializer = tf.random_uniform_initializer(minval = -1e-3, maxval = 1e-3))
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