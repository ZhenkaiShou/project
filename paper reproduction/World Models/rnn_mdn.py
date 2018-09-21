import math
import tensorflow as tf

import config

# Local hyperparameters
Z_LENGTH = config.Z_LENGTH
A_LENGTH = config.A_LENGTH
HIDDEN_UNITS = config.HIDDEN_UNITS
MODES = config.MODES
EPSILON = 1e-8
MAX_GRAD = 1.0

class RNN_MDN(object):
  def __init__(self):
    pass
  
  def build_model(self, is_training = False, is_assigning = False, is_single_input = False):
    self.graph = tf.Graph()
    with self.graph.as_default():
      if is_single_input:
        self.ZA = tf.placeholder(tf.float32, (1, 1, Z_LENGTH + A_LENGTH))
      else:
        self.ZA = tf.placeholder(tf.float32, (None, None, Z_LENGTH + A_LENGTH))
      self.Target_Z = tf.placeholder(tf.float32, (None, None, Z_LENGTH))
      
      # LSTM.
      cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_UNITS)
      if is_single_input:
        self.initial_state = cell.zero_state(1, tf.float32)
      else:
        self.initial_state = cell.zero_state(tf.shape(self.ZA)[0], tf.float32)
      
      # rnn_output = a stack of hidden units for all time sequences.
      # finat_state = a tuple of the memory cell and hidden units for the last sequence.
      # shape(rnn_output) = (batch_size, time_length, HIDDEN_UNITS)
      rnn_output, self.final_state = tf.nn.dynamic_rnn(cell, self.ZA, initial_state = self.initial_state, scope = "rnn")
      
      # MDN.
      # shape(rnn_output) = (batch_size * time_length, HIDDEN_UNITS)
      rnn_output = tf.reshape(rnn_output, (-1, HIDDEN_UNITS))
      # pi(X) = probability of X belonging to a certain mode (s.t. sum up to 1)
      # mu(X) = mean value of the output for that mode
      # sigma(X) = standard deviation of the output for that mode (s.t. sigma > 0)
      # shape(pi) = (batch_size * time_length * Z_LENGTH, MODES)
      self.logits = tf.reshape(tf.layers.dense(rnn_output, Z_LENGTH * MODES, name = "mdn_logits"), (-1, MODES))
      self.pi = tf.nn.softmax(self.logits)
      self.mu = tf.reshape(tf.layers.dense(rnn_output, Z_LENGTH * MODES, name = "mdn_mu"), (-1, MODES))
      log_sigma = tf.reshape(tf.layers.dense(rnn_output, Z_LENGTH * MODES, name = "mdn_log_sigma"), (-1, MODES))
      self.sigma = tf.exp(log_sigma)
      
      # Loss function.
      gaussian = 1 / (self.sigma * tf.sqrt(2 * math.pi)) * tf.exp(-0.5 * tf.square((tf.reshape(self.Target_Z, (-1, 1)) - self.mu) / self.sigma))
      self.loss = tf.reduce_mean(-tf.log(tf.reduce_sum(self.pi * gaussian, -1) + EPSILON))
      
      if is_training:
        # Optimization.
        self.LR = tf.placeholder(tf.float32, ())
        
        optimizer = tf.train.AdamOptimizer(self.LR)
        gradients = optimizer.compute_gradients(self.loss)
        grad_clip = [(tf.clip_by_value(grad, -MAX_GRAD, MAX_GRAD), var) for grad, var in gradients]
        self.train_op = optimizer.apply_gradients(grad_clip)
      
      if is_assigning:
        # Assign operations for generating random networks.
        self.Assigned_Value = tf.placeholder(tf.float32)
        
        self.assign_op = [tf.assign(var, self.Assigned_Value) for var in tf.trainable_variables()]