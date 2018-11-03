import math
import tensorflow as tf

import config

# Local hyperparameters
Z_LENGTH = config.Z_LENGTH
A_LENGTH = config.A_LENGTH
H_LENGTH = config.H_LENGTH
C_L2 = 1e-5

class RNN(object):
  def __init__(self, name = "rnn", is_single_input = False):
    self.name = name
    self.build_model(is_single_input)
  
  def build_model(self, is_single_input):
    with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
      if is_single_input:
        self.Z = tf.placeholder(tf.float32, (1, 1, Z_LENGTH))
        self.A = tf.placeholder(tf.float32, (1, 1, A_LENGTH))
      else:
        self.Z = tf.placeholder(tf.float32, (None, None, Z_LENGTH))
        self.A = tf.placeholder(tf.float32, (None, None, A_LENGTH))
      za = tf.concat([self.Z, self.A], 2)
      self.Target_Z = tf.placeholder(tf.float32, (None, None, Z_LENGTH))
      
      # LSTM.
      cell = tf.contrib.rnn.BasicLSTMCell(H_LENGTH)
      if is_single_input:
        self.initial_state = cell.zero_state(1, tf.float32)
      else:
        self.initial_state = cell.zero_state(tf.shape(za)[0], tf.float32)
      
      # rnn_output = a stack of hidden units for all time sequences.
      # finat_state = a tuple of the memory cell and hidden units for the last sequence.
      # shape(rnn_output) = (batch_size, time_length, H_LENGTH)
      rnn_output, self.final_state = tf.nn.dynamic_rnn(cell, za, initial_state = self.initial_state)
      
      # A dense layer to predict the next state.
      self.z_next = tf.layers.dense(rnn_output, Z_LENGTH)
    self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.name)
    
    # Loss function.
    self.mse_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.z_next - self.Target_Z), [1, 2]))
    self.regularization_loss = C_L2 * sum([2 * tf.nn.l2_loss(var) for var in self.trainable_variables])
    self.loss = self.mse_loss + self.regularization_loss
    
  def build_training(self):
    # Optimization.
    self.LR = tf.placeholder(tf.float32, ())
    optimizer = tf.train.AdamOptimizer(self.LR)
    self.train_op = optimizer.minimize(self.loss, var_list = self.trainable_variables)