import multiprocessing
import numpy as np
import os

import config
from vae import VAE

# Local hyperparameters
PROCESS = 1

SAVE_VAE_DIR = config.SAVE_VAE_DIR
RAW_DATA_DIR = config.RAW_DATA_DIR
ENCODING_DATA_DIR = config.ENCODING_DATA_DIR

def encoding(file_name = "vae"):
  # Create folder.
  if not os.path.isdir(ENCODING_DATA_DIR):
    os.makedirs(ENCODING_DATA_DIR)
  
  file_list = os.listdir(RAW_DATA_DIR)
  data_length = len(file_list)
  
  # Determine how many files to encode for each process.
  start_index = 0
  base = data_length // PROCESS
  reminder = data_length % PROCESS
  arg_list = [None for _ in range(PROCESS)]
  for i in range(PROCESS):
    num_files = base + 1 if i < reminder else base
    arg_list[i] = [file_name, file_list[start_index:start_index + num_files]]
    start_index += num_files
  
  with multiprocessing.Pool(PROCESS) as p:
    p.starmap(encoding_process, arg_list)

def encoding_process(file_name, file_list):
  import tensorflow as tf
  
  file_name_list = [os.path.splitext(file)[0] for file in file_list]
  data_length = len(file_list)

  # Load models.
  vae = VAE(name = "vae")
  
  with tf.Session() as sess:
    # Load variables.
    saver = tf.train.Saver()
    saver.restore(sess, SAVE_VAE_DIR + file_name)
    
    for i in range(data_length):
      # Load raw data.
      data = np.load(RAW_DATA_DIR + file_list[i])
      obs = data["obs"]
      action = data["action"]
      
      # Compute the mean.
      z = sess.run(vae.mu, feed_dict = {vae.Input: obs / 255.0})
      
      # Save file.
      np.savez_compressed(ENCODING_DATA_DIR + file_name_list[i], z = z, action = action)

if __name__ == "__main__":
  encoding()