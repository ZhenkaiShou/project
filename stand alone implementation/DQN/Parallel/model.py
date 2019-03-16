import tensorflow as tf

from config import *

class QValueNetwork(object):
  def __init__(self, obs_space, action_space, name, auxiliary_network = None):
    self.obs_space = obs_space
    self.action_space = action_space
    self.name = name
    
    self.Obs = tf.placeholder(tf.float32, (None, *self.obs_space.shape))
    
    with tf.variable_scope(self.name):
      # Encode the observation.
      feature = encode(self.Obs)
      h = tf.layers.dense(feature, units = 512, use_bias = True, activation = tf.nn.relu)
      # Duel network into: expectation and advantage.
      # shape(expectation) = (batch_size, 1)
      exp = tf.layers.dense(h, units = 512, use_bias = True, activation = tf.nn.relu)
      exp = tf.layers.dense(h, units = 1, use_bias = True)
      # shape(advantage) = (batch_size, num_action)
      adv = tf.layers.dense(h, units = 512, use_bias = True, activation = tf.nn.relu)
      adv = tf.layers.dense(h, units = self.action_space.n, use_bias = True)
      # q(s, a) = exp(s) + adv(s, a)
      # shape(q) = (batch_size, num_action)
      self.q = exp + adv - tf.reduce_mean(adv, axis = 1, keepdims = True)
    self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.name)
    self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = self.name)
    
    # Loss function.
    self.Action = tf.placeholder(tf.int32, (None,))
    self.TargetQ = tf.placeholder(tf.float32, (None,))
    q_a = tf.reduce_sum(self.q * tf.one_hot(self.Action, self.action_space.n), 1)
    self.loss = tf.reduce_mean(tf.square(self.TargetQ - q_a))
    
    # Optimization.
    self.LR = tf.placeholder(tf.float32, ())
    optimizer = tf.train.AdamOptimizer(self.LR)
    self.global_step = tf.train.get_or_create_global_step()
    self.train_op = optimizer.minimize(self.loss, var_list = self.trainable_variables, global_step = self.global_step)
    
    # Update operation.
    if auxiliary_network is not None:
      # Synchronize the network (self network <- auxiliary network).
      self.sync_op = [tf.assign(ref, value) for ref, value in zip(self.variables, auxiliary_network.variables)]

def encode(input):
  h = tf.layers.conv2d(input, filters = 32, kernel_size = (8, 8), strides = (4, 4), use_bias = True, activation = tf.nn.relu)
  h = tf.layers.conv2d(h, filters = 64, kernel_size = (4, 4), strides = (2, 2), use_bias = True, activation = tf.nn.relu)
  h = tf.layers.conv2d(h, filters = 64, kernel_size = (3, 3), strides = (1, 1), use_bias = True, activation = tf.nn.relu)
  feature = tf.layers.flatten(h)
  return feature