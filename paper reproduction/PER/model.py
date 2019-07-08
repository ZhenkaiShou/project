import tensorflow as tf

class QValueNetwork(object):
  def __init__(self, input_obs, num_action, network_type, name):
    self.name = name
    
    with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
      # Get feature.
      if network_type == "conv":
        feature = conv(input_obs)
      elif network_type == "dense":
        feature = dense(input_obs)
      else:
        raise NotImplementedError
      
      # Duel network into: expectation and advantage.
      # shape(expectation) = (batch_size, 1)
      exp = feature
      exp = tf.layers.dense(exp, units = 256, activation = tf.nn.relu)
      exp = tf.layers.dense(exp, units = 1)
      # shape(advantage) = (batch_size, num_action)
      adv = feature
      adv = tf.layers.dense(adv, units = 256, activation = tf.nn.relu)
      adv = tf.layers.dense(adv, units = num_action)
      adv = adv - tf.reduce_mean(adv, axis = 1, keepdims = True)
      # q(s, a) = exp(s) + adv(s, a)
      # shape(q) = (batch_size, num_action)
      self.q = exp + adv
    self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.name)
    self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = self.name)

def dense(input):
  h = input
  h = tf.layers.dense(h, units = 64, activation = tf.tanh)
  h = tf.layers.dense(h, units = 64, activation = tf.tanh)
  return h

def conv(input):
  h = input
  h = tf.layers.conv2d(h, filters = 32, kernel_size = (8, 8), strides = (4, 4), activation = tf.nn.relu)
  h = tf.layers.conv2d(h, filters = 64, kernel_size = (4, 4), strides = (2, 2), activation = tf.nn.relu)
  h = tf.layers.conv2d(h, filters = 64, kernel_size = (3, 3), strides = (1, 1), activation = tf.nn.relu)
  h = tf.layers.flatten(h)
  return h