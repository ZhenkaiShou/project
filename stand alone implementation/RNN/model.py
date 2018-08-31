import tensorflow as tf
from config import * 

class RNN_Model(object):
  def rnn(self):
    # Basic Recurrent Neural Network implementation.
    
    # Number of training variables
    # RNN, kernel: (128+28)*128
    # RNN, bias: 128
    # Dense, kernel: 128*10 
    # Total: 21376
    self.Inputs = tf.placeholder(tf.float32, (None, TIME_LENGTH, INPUT_LENGTH))
    self.Labels = tf.placeholder(tf.float32, (None, 10))
    self.LR = tf.placeholder(tf.float32, ())
    
    cell = tf.contrib.rnn.BasicRNNCell(FEATURE_NUM)
    initial_state = cell.zero_state(tf.shape(self.Inputs)[0], tf.float32)
    # outputs = a stack of hidden units for all time sequences.
    # state = the hidden units for the last sequence.
    self.outputs, self.state = tf.nn.dynamic_rnn(cell, self.Inputs, tf.tile([TIME_LENGTH], [tf.shape(self.Inputs)[0]]), initial_state)
    self.logits = tf.layers.dense(self.state, 10, use_bias = False)
    self.y = tf.nn.softmax(self.logits)
    
    self.loss = tf.losses.softmax_cross_entropy(self.Labels, self.logits)
    
    # Add L2 regularization loss.
    self.regularization_loss = sum(C_L2 * 2 * tf.nn.l2_loss(var) for var in tf.trainable_variables() if "kernel" in var.name)
    tf.losses.add_loss(self.regularization_loss)
    self.total_loss = tf.losses.get_total_loss()
    
    # Optimization.
    optimizer = tf.train.GradientDescentOptimizer(self.LR)
    self.train_op = optimizer.minimize(self.total_loss)
  
  def lstm(self):
    # Long Short-Term Memory Network implementation.
    
    # Number of training variables
    # LSTM, kernel: (128+28)*128*4 (1 memory cell + 3 gates)
    # LSTM, bias: 128*4 (1 memory cell + 3 gates)
    # Dense, kernel: 128*10 
    # Total: 81664
    self.Inputs = tf.placeholder(tf.float32, (None, TIME_LENGTH, INPUT_LENGTH))
    self.Labels = tf.placeholder(tf.float32, (None, 10))
    self.LR = tf.placeholder(tf.float32, ())
    
    cell = tf.contrib.rnn.BasicLSTMCell(FEATURE_NUM)
    initial_state = cell.zero_state(tf.shape(self.Inputs)[0], tf.float32)
    # outputs = a stack of hidden units for all time sequences.
    # state = a tuple of the memory cell and hidden units for the last sequence.
    self.outputs, self.state = tf.nn.dynamic_rnn(cell, self.Inputs, tf.tile([TIME_LENGTH], [tf.shape(self.Inputs)[0]]), initial_state)
    self.logits = tf.layers.dense(self.state[1], 10, use_bias = False)
    self.y = tf.nn.softmax(self.logits)
    
    self.loss = tf.losses.softmax_cross_entropy(self.Labels, self.logits)
    
    # Add L2 regularization loss.
    self.regularization_loss = sum(C_L2 * 2 * tf.nn.l2_loss(var) for var in tf.trainable_variables() if "kernel" in var.name)
    tf.losses.add_loss(self.regularization_loss)
    self.total_loss = tf.losses.get_total_loss()
    
    # Optimization.
    optimizer = tf.train.GradientDescentOptimizer(self.LR)
    self.train_op = optimizer.minimize(self.total_loss)
  
  def fc(self):
    # A single-hidden-layer network for comparison.
    
    # Number of training variables
    # Hidden layer, kernel: 28*28*128
    # Hidden layer, bias: 128
    # Dense, kernel: 128*10
    # Total: 101760
    self.Inputs = tf.placeholder(tf.float32, (None, TIME_LENGTH, INPUT_LENGTH))
    self.Labels = tf.placeholder(tf.float32, (None, 10))
    self.LR = tf.placeholder(tf.float32, ())
    
    self.inputs = tf.reshape(self.Inputs, (-1, TIME_LENGTH * INPUT_LENGTH))
    self.z = tf.layers.dense(self.inputs, FEATURE_NUM, activation = tf.nn.relu, use_bias = True)
    self.logits = tf.layers.dense(self.z, 10, use_bias = False)
    self.y = tf.nn.softmax(self.logits)
    
    self.loss = tf.losses.softmax_cross_entropy(self.Labels, self.logits)
    
    # Add L2 regularization loss.
    self.regularization_loss = sum(C_L2 * 2 * tf.nn.l2_loss(var) for var in tf.trainable_variables() if "kernel" in var.name)
    tf.losses.add_loss(self.regularization_loss)
    self.total_loss = tf.losses.get_total_loss()
    
    # Optimization.
    optimizer = tf.train.GradientDescentOptimizer(self.LR)
    self.train_op = optimizer.minimize(self.total_loss)
