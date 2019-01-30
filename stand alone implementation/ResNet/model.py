import numpy as np
import tensorflow as tf

from config import *

class Model(object):
  def __init__(self, hps):
    self.hps = hps
    self.name = "model"
    self.IsTraining = tf.placeholder(tf.bool, ())
    self.Input = tf.placeholder(tf.float32, (None, IMAGE_LENGTH, IMAGE_LENGTH, 3))
    self.Label = tf.placeholder(tf.int32, (None,))
    
    input = self.Input
    # Data augmentation during training.
    if self.hps["flip_data"]:
      input = tf.cond(self.IsTraining, true_fn = lambda: tf.image.random_flip_left_right(input), false_fn = lambda: input)
    if self.hps["crop_data"]:
      input = tf.cond(self.IsTraining, true_fn = lambda: tf.image.resize_image_with_crop_or_pad(input, IMAGE_LENGTH + 8, IMAGE_LENGTH + 8), false_fn = lambda: input)
      crop_fn = tf.map_fn(lambda data: tf.image.random_crop(data, size = [IMAGE_LENGTH, IMAGE_LENGTH, 3]), input)
      input = tf.cond(self.IsTraining, true_fn = lambda: crop_fn, false_fn = lambda: input)
    input.set_shape([None, IMAGE_LENGTH, IMAGE_LENGTH, 3])
    
    with tf.variable_scope(self.name):
      # First convolutional block: (None, 32, 32, 3) -> (None, 32, 32, 32).
      h = self.conv_block(input, filters = 32)
      
      if self.hps["network_type"] == "Res4":
        # Followed by 4 residual blocks.
        # (None, 32, 32, 32) -> (None, 32, 32, 32)
        h = self.residual_block(h, filters = 32, first_conv_strides = 1)
        # (None, 32, 32, 32) -> (None, 16, 16, 64)
        h = self.residual_block(h, filters = 64, first_conv_strides = 2)
        # (None, 16, 16, 64) -> (None, 8, 8, 128)
        h = self.residual_block(h, filters = 128, first_conv_strides = 2)
        # (None, 8, 8, 128) -> (None, 4, 4, 256)
        h = self.residual_block(h, filters = 256, first_conv_strides = 2)
      elif self.hps["network_type"] == "Conv8":
        # Followed by 8 convolutional blocks.
        # (None, 32, 32, 32) -> (None, 32, 32, 32)
        h = self.conv_block(h, filters = 32)
        h = self.conv_block(h, filters = 32)
        # (None, 32, 32, 32) -> (None, 16, 16, 64)
        h = self.conv_block(h, filters = 64, strides = 2)
        h = self.conv_block(h, filters = 64)
        # (None, 16, 16, 64) -> (None, 8, 8, 128)
        h = self.conv_block(h, filters = 128, strides = 2)
        h = self.conv_block(h, filters = 128)
        # (None, 8, 8, 128) -> (None, 4, 4, 256)
        h = self.conv_block(h, filters = 256, strides = 2)
        h = self.conv_block(h, filters = 256)
      elif self.hps["network_type"] == "None":
        pass
      else:
        raise NotImplementedError
      
      if self.hps["global_average_pool"]:
        # Global average pooling: (None, 4, 4, 256) -> (None, 1, 1, 256).
        if self.hps["network_type"] == "Res4" or self.hps["network_type"] == "Conv8":
          spatial_size = IMAGE_LENGTH
          for _ in range(3):
            spatial_size = np.ceil(spatial_size / 2)
        elif self.hps["network_type"] == "None":
          spatial_size = IMAGE_LENGTH
        h = tf.layers.average_pooling2d(h, pool_size = (spatial_size, spatial_size), strides = (1, 1), padding = "valid")
      
      # Fully connected layer to predict the class.
      h = tf.layers.flatten(h)
      logits = tf.layers.dense(h, NUM_CLASS, use_bias = False)
      self.prediction = tf.argmax(logits, 1, output_type = tf.int32)
      self.wrong_pred = tf.reduce_sum(tf.cast(tf.not_equal(self.prediction, self.Label), tf.float32))
    self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.name)
    
    # Loss function.
    cross_entropy_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.Label, logits = logits))
    regularization_loss = self.hps["c_l2"] * 2 * sum(tf.nn.l2_loss(var) for var in self.trainable_variables if "kernel" in var.name)
    total_loss = cross_entropy_loss + regularization_loss
    self.loss = cross_entropy_loss
    
    # Optimization.
    self.LR = tf.placeholder(tf.float32, ())
    if self.hps["optimizer"] == "Momentum":
      optimizer = tf.train.MomentumOptimizer(self.LR, momentum = 0.9)
    elif self.hps["optimizer"] == "Adam":
      optimizer = tf.train.AdamOptimizer(self.LR)
    else:
      raise NotImplementedError
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      self.train_op = optimizer.minimize(total_loss)
  
  def conv_block(self, input, filters, kernel = 3, strides = 1):
    # Convolution block: 
    #  1. 2D-Convolution
    #  2. Dropout
    #  3. Batch normalization
    #  4. ReLU
    h = tf.layers.conv2d(input, filters = filters, kernel_size = (kernel, kernel), strides = (strides, strides), use_bias = False, padding = "same")
    h = tf.layers.dropout(h, self.hps["dropout_rate"], training = self.IsTraining)
    if self.hps["batch_norm"]:
      h = tf.layers.batch_normalization(h, training = self.IsTraining)
    h = tf.nn.relu(h)
    return h
  
  def residual_block(self, input, filters, kernel = 3, first_conv_strides = 1):
    # Residual block:
    #  1. 2D-Convolution
    #  2. Dropout
    #  3. Batch normalization
    #  4. ReLU
    #  5. 2D-Convolution
    #  6. Dropout
    #  7. Batch normalization
    #  8. Skip connection
    #  9. ReLU
    h = self.conv_block(input, filters, kernel = kernel, strides = first_conv_strides)
    h = tf.layers.conv2d(h, filters = filters, kernel_size = (kernel, kernel), strides = (1, 1), use_bias = False, padding = "same")
    h = tf.layers.dropout(h, self.hps["dropout_rate"], training = self.IsTraining)
    if self.hps["batch_norm"]:
      h = tf.layers.batch_normalization(h, training = self.IsTraining)
    if first_conv_strides > 1:
      shortcut = tf.layers.conv2d(input, filters = filters, kernel_size = (first_conv_strides, first_conv_strides), strides = (first_conv_strides, first_conv_strides), use_bias = False, padding = "same")
      shortcut = tf.layers.dropout(shortcut, self.hps["dropout_rate"], training = self.IsTraining)
      if self.hps["batch_norm"]:
        shortcut = tf.layers.batch_normalization(shortcut, training = self.IsTraining)
    else:
      shortcut = input
    h = shortcut + h
    h = tf.nn.relu(h)
    return h