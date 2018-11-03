import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

import config
from rnn import RNN

# Local hyperparameters
MAX_FRAME = config.MAX_FRAME
Z_LENGTH = config.Z_LENGTH
EPOCH = 80
BATCH_SIZE = 100
LEARNING_RATE = 1e-3

ENCODING_DATA_DIR = config.ENCODING_DATA_DIR
SAVE_RNN_DIR = config.SAVE_RNN_DIR
FIGURE_TRAINING_DIR = config.FIGURE_TRAINING_DIR

def rnn_training(file_name = "rnn"):
  # Create folders.
  if not os.path.isdir(SAVE_RNN_DIR):
    os.makedirs(SAVE_RNN_DIR)
  if not os.path.isdir(FIGURE_TRAINING_DIR):
    os.makedirs(FIGURE_TRAINING_DIR)
  
  file_list = os.listdir(ENCODING_DATA_DIR)
  data_length = len(file_list)
  num_batch = data_length // BATCH_SIZE
  num_iter = EPOCH * num_batch
  
  # Load data.
  list_z = []
  list_action = []
  for i in range(data_length):
    data = np.load(ENCODING_DATA_DIR + file_list[i])
    list_z.append(data["z"])
    list_action.append(data["action"])
  list_z = np.array(list_z)
  list_action = np.array(list_action)
  
  # Load models.
  rnn = RNN(name = "rnn", is_single_input = False)
  rnn.build_training()
  initialize_op = tf.global_variables_initializer()
  
  with tf.Session() as sess:
    # Initialize the network.
    sess.run(initialize_op)
  
    list_loss = []
    
    for epoch in range(EPOCH):
      # Shuffle training data.
      random_index = np.random.permutation(data_length)
      
      for batch in range(num_batch):
        # Define input and target data.
        batch_index = random_index[batch * BATCH_SIZE:(batch+1) * BATCH_SIZE]
        z = list_z[batch_index]
        a = list_action[batch_index]
        
        _, mse_loss = sess.run([rnn.train_op, rnn.mse_loss], feed_dict = {rnn.Z: z[:, :-1, :], rnn.A: a[:, :-1, :], rnn.Target_Z: z[:, 1:, :], rnn.LR: LEARNING_RATE})
        list_loss.append(mse_loss)
        
        if (epoch * num_batch + batch) % 5 == 0:
          print("Iteration ", format(epoch * num_batch + batch, "04d"), ":", sep = "") 
          print("  Loss = ", format(mse_loss, ".8f"), sep = "")
    
    # Save the parameters.
    saver = tf.train.Saver()
    saver.save(sess, SAVE_RNN_DIR + file_name)
  tf.contrib.keras.backend.clear_session()
  
  # Plot the training loss.
  list_iter = list(range(num_iter))
  f, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (5, 5))
  ax.plot(list_iter, list_loss, "r-", label = "Loss")
  ax.set_title("Training Loss")
  ax.set_xlabel("Iteration")
  ax.set_ylabel("Loss")
  ax.legend(loc = "upper right")
  ax.ticklabel_format(style = "sci", axis = "x", scilimits = (0, 0))
  ax.grid()
  
  f.savefig(FIGURE_TRAINING_DIR + file_name + ".png")
  plt.close(f)

if __name__ == "__main__":
  rnn_training()