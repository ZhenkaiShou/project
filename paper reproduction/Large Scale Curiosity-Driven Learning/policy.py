import numpy as np
import tensorflow as tf

import config

# Local hyperparameters
HIDDEN_UNIT = config.HIDDEN_UNIT
LEARNING_RATE = config.LEARNING_RATE
CLIP_RANGE = 0.2
COEF_ENTROPY = 1e-3

def norm_initializer(shape, dtype = None, partition_info = None):
  value = np.random.randn(*shape).astype(np.float32)
  norm = np.sqrt(np.sum(np.square(value), axis = 0, keepdims = True))
  norm_value = value / norm
  return tf.constant(norm_value)

class Policy(object):
  def __init__(self, obs_space, action_space, is_training = True):
    self.scope = "policy"
    self.obs_space = obs_space
    self.action_space = action_space
    
    with tf.variable_scope(self.scope):      
      self.Obs = tf.placeholder(tf.float32, (None, None, *self.obs_space.shape))
      self.VTarget = tf.placeholder(tf.float32, (None, None))
      self.Action = tf.placeholder(tf.int32, (None, None))
      self.LogProbOld = tf.placeholder(tf.float32, (None, None))
      self.Adv = tf.placeholder(tf.float32, (None, None))
      
      # shape(z) = (batch_size, time_length, hidden_unit)
      z = self.encode(self.Obs)
      
      # Additional layers before split into policy and value heads.
      z = tf.layers.dense(z, HIDDEN_UNIT, activation = tf.nn.relu, kernel_initializer = norm_initializer)
      z = tf.layers.dense(z, HIDDEN_UNIT, activation = tf.nn.relu, kernel_initializer = norm_initializer)
      
      # Value head: v = value(z).
      v = tf.layers.dense(z, 1, use_bias = False, kernel_initializer = norm_initializer)
      # shape(v) = (batch_size, time_length)
      self.v = tf.squeeze(v, 2)
      
      # Policy head: pi = policy(z).
      # shape(pi_logits) = (batch_size, time_length, num_action)
      pi_logits = tf.layers.dense(z, self.action_space.n, use_bias = False, kernel_initializer = norm_initializer)
      self.action = tf.argmax(pi_logits, 2, output_type = tf.int32)
      
      # Get distribution.
      pi_distribution = tf.distributions.Categorical(pi_logits)
      
      # Sample action with Gumbel softmax trick.
      u = tf.random_uniform(tf.shape(pi_logits))
      self.sampled_action = tf.argmax(pi_logits - tf.log(-tf.log(u)), 2, output_type = tf.int32)
      self.sampled_log_prob = pi_distribution.log_prob(self.sampled_action)
      
      # Value loss.
      self.value_loss = tf.reduce_mean(tf.square(self.v - self.VTarget))
      # Policy loss.
      ratio = tf.exp(pi_distribution.log_prob(self.Action) - self.LogProbOld)
      pg_loss1 = ratio * self.Adv
      pg_loss2 = tf.clip_by_value(ratio, 1 - CLIP_RANGE, 1 + CLIP_RANGE) * self.Adv
      self.pg_loss = tf.reduce_mean(-tf.minimum(pg_loss1, pg_loss2))
      
      # Entropy loss.
      self.entropy_loss = -pi_distribution.entropy()
      # Total loss.
      self.loss = self.value_loss + self.pg_loss + COEF_ENTROPY * self.entropy_loss
    
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