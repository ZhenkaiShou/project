import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

import config
from vae import VAE

# Local hyperparameters
EPOCH = 10
BATCH_SIZE = 100
MAX_EPISODE = config.MAX_EPISODE
MAX_FRAME = config.MAX_FRAME
LEARNING_RATE = 1e-4

RAW_DATA_DIR = config.RAW_DATA_DIR
SAVE_VAE_DIR = config.SAVE_VAE_DIR
CSV_DIR = config.CSV_DIR
FIGURE_TRAINING_DIR = config.FIGURE_TRAINING_DIR

def vae_training(file_name = "vae"):
  # Create folders.
  if not os.path.isdir(SAVE_VAE_DIR):
    os.makedirs(SAVE_VAE_DIR)
  if not os.path.isdir(CSV_DIR):
    os.makedirs(CSV_DIR)
  if not os.path.isdir(FIGURE_TRAINING_DIR):
    os.makedirs(FIGURE_TRAINING_DIR)
  
  file_list = os.listdir(RAW_DATA_DIR)
  data_length = len(file_list)
  num_batch = data_length * MAX_FRAME // BATCH_SIZE
  num_iter = EPOCH * num_batch
  
  # Load data.
  list_obs = []
  for i in range(data_length):
    obs = np.load(RAW_DATA_DIR + file_list[i])["obs"]
    list_obs.append(np.reshape(obs, (-1, 64, 64, 3)))
  list_obs = np.concatenate(list_obs, 0)
  
  # Load models.
  vae = VAE()
  vae.build_model(is_training = True, is_assigning = False)
  
  with tf.Session(graph = vae.graph) as sess:
    # Initialize the network.
    sess.run(tf.global_variables_initializer())
    
    list_reconstruction_loss = []
    list_kl_divergence = []
    list_loss = []
    
    for epoch in range(EPOCH):
      # Shuffle training data.
      np.random.shuffle(list_obs)
      
      for batch in range(num_batch):
        obs = list_obs[batch * BATCH_SIZE: (batch+1) * BATCH_SIZE]
        _, reconstruction_loss, kl_divergence, loss = sess.run([vae.train_op, vae.reconstruction_loss, vae.kl_divergence, vae.loss], feed_dict = {vae.Input: obs / 255.0, vae.LR: LEARNING_RATE})
        
        list_reconstruction_loss.append(reconstruction_loss)
        list_kl_divergence.append(kl_divergence)
        list_loss.append(loss)
        
        if (epoch * num_batch + batch) % 100 == 0:
          print("Iteration ", format(epoch * num_batch + batch, "06d"), ":", sep = "") 
          print("  Reconstruction Loss = ", format(reconstruction_loss, ".8f"), ", KL Divergence = ", format(kl_divergence, ".8f"), sep = "")
    
    # Save the parameters.
    saver = tf.train.Saver()
    saver.save(sess, SAVE_VAE_DIR + file_name)
  tf.contrib.keras.backend.clear_session()
  
  # Store data in the csv file.
  with open(CSV_DIR + file_name + ".csv", "w") as f:
    fieldnames = ["Iteration", "Reconstruction Loss", "KL Divergence", "Loss"]
    writer = csv.DictWriter(f, fieldnames = fieldnames, lineterminator = "\n")
    writer.writeheader()
    for iter in range(num_iter):
      content = {"Iteration": iter, "Reconstruction Loss": list_reconstruction_loss[iter], "KL Divergence": list_kl_divergence[iter], "Loss": list_loss[iter]}
      writer.writerow(content)
  
  # Plot the training loss.
  list_iter = list(range(num_iter))
  f, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (5, 5))
  ax.plot(list_iter, list_reconstruction_loss, "r-", label = "Reconstruction Loss")
  ax.plot(list_iter, list_kl_divergence, "b-", label = "KL Divergence")
  ax.set_title("Training Loss")
  ax.set_xlabel("Iteration")
  ax.set_ylabel("Loss")
  ax.legend(loc = "upper right")
  ax.ticklabel_format(style = "sci", axis = "x", scilimits = (0, 0))
  ax.grid()
  
  f.savefig(FIGURE_TRAINING_DIR + file_name + ".png")
  plt.close(f)

if __name__ == "__main__":
  vae_training()