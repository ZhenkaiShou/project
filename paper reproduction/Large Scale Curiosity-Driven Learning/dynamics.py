import numpy as np
import tensorflow as tf

import config

# Local hyperparameters
HIDDEN_UNIT = config.HIDDEN_UNIT
LEARNING_RATE = config.LEARNING_RATE
RESIDUAL_BLOCK = 4

def norm_initializer(shape, dtype = None, partition_info = None):
  value = np.random.randn(*shape).astype(np.float32)
  norm = np.sqrt(np.sum(np.square(value), axis = 0, keepdims = True))
  norm_value = value / norm
  return tf.constant(norm_value)

class Dynamics(object):
  def __init__(self, obs_space, action_space, auxiliary_task, is_training = True):
    self.scope = "dynamics"
    self.obs_space = obs_space
    self.action_space = action_space
    
    with tf.variable_scope(self.scope):
      self.Obs = tf.placeholder(tf.float32, (None, None, *self.obs_space.shape))
      self.ObsNext = tf.placeholder(tf.float32, (None, None, *self.obs_space.shape))
      self.Action = tf.placeholder(tf.int32, (None, None))
      
      # shape(z) = (batch_size, time_length, hidden_unit)
      self.z = self.encode(self.Obs)
      self.z_next = self.encode(self.ObsNext)
      # shape(action_onehot) = (batch_size, time_length, num_action)
      action_onehot = tf.one_hot(self.Action, self.action_space.n)
      
      def concat_action(input):
        return tf.concat([input, action_onehot], 2)
      
      # Dynamics: z_next = dynamics(z, action).
      input = concat_action(tf.stop_gradient(self.z))
      hidden = tf.layers.dense(input, HIDDEN_UNIT, activation = tf.nn.leaky_relu)
      for _ in range(RESIDUAL_BLOCK):
        residual = tf.layers.dense(concat_action(hidden), HIDDEN_UNIT, activation = tf.nn.leaky_relu)
        residual = tf.layers.dense(concat_action(residual), HIDDEN_UNIT)
        hidden = hidden + residual
      # shape(z_next) = (batch_size, time_length, hidden_unit)
      z_next = tf.layers.dense(concat_action(hidden), HIDDEN_UNIT)
      
      # Intrinsic reward.
      self.intrinsic_reward = tf.reduce_mean(tf.square(z_next - tf.stop_gradient(self.z_next)), 2)
      
      # Dynamics loss function.
      self.dyna_loss = tf.reduce_mean(self.intrinsic_reward)
      
      # Auxiliary task and loss function.
      if auxiliary_task == "random_features":
        self.auxiliary_loss = self.random_features()
      elif auxiliary_task == "inverse_dynamics":
        self.auxiliary_loss = self.inverse_dynamics()
      else:
        raise NotImplementedError
      
      self.loss = self.dyna_loss + self.auxiliary_loss
    
    self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.scope)
    
    if is_training:
      # Optimization.
      optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
      self.train_op = optimizer.minimize(self.loss, var_list = self.trainable_variables)
  
  def encode(self, input):
    # Encode the observation: z = encode(obs).
    with tf.variable_scope("encoding", reuse = tf.AUTO_REUSE):
      input_shape = tf.shape(input)
      z = tf.reshape(input, (-1, *self.obs_space.shape))
      z = tf.layers.conv2d(z, 32, kernel_size = (8, 8), strides = (4, 4), activation = tf.nn.leaky_relu)
      z = tf.layers.conv2d(z, 64, kernel_size = (4, 4), strides = (2, 2), activation = tf.nn.leaky_relu)
      z = tf.layers.conv2d(z, 64, kernel_size = (3, 3), strides = (1, 1), activation = tf.nn.leaky_relu)
      z = tf.reshape(z, (input_shape[0], input_shape[1], np.prod(z.get_shape().as_list()[1:])))
      z = tf.layers.dense(z, HIDDEN_UNIT, kernel_initializer = norm_initializer)
    return z
  
  def inverse_dynamics(self):
    with tf.variable_scope("inverse_dynamics"):
      # Inverse dynamics: action = inverse_dynamics(z, z_next).
      input = tf.concat([self.z, self.z_next], 2)
      hidden = tf.layers.dense(input, HIDDEN_UNIT, activation = tf.nn.relu, kernel_initializer = norm_initializer)
      # shape(logits) = (batch_size, time_length, num_action)
      logits = tf.layers.dense(hidden, self.action_space.n, use_bias = False, kernel_initializer = norm_initializer)
      loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.Action, logits = logits))
    return loss
  
  def random_features(self):
    # Just keep the encoding network parameters fixed.
    loss = tf.zeros((), tf.float32)
    return loss