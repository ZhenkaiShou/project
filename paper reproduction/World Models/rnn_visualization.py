import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

import config
from rnn_mdn import RNN_MDN
from vae import VAE

# Local hyperparameters
MAX_FRAME = config.MAX_FRAME
Z_LENGTH = config.Z_LENGTH
A_LENGTH = config.A_LENGTH
MODES = config.MODES

SAVE_VAE_DIR = config.SAVE_VAE_DIR
SAVE_RNN_DIR = config.SAVE_RNN_DIR
RAW_DATA_DIR = config.RAW_DATA_DIR
ENCODING_DATA_DIR = config.ENCODING_DATA_DIR
FIGURE_RNN_VISUALIZATION_DIR = config.FIGURE_RNN_VISUALIZATION_DIR

def rnn_visualization(temperature = 1.0, vae_file_name = "vae", rnn_file_name = "rnn", random_file = None):
  # Create folders.
  if not os.path.isdir(FIGURE_RNN_VISUALIZATION_DIR):
    os.makedirs(FIGURE_RNN_VISUALIZATION_DIR)
  
  if random_file == None:
    # Load random data.
    file_list = os.listdir(ENCODING_DATA_DIR)
    random_file = np.random.choice(file_list)
  random_file_name = os.path.splitext(random_file)[0]
  obs = np.load(RAW_DATA_DIR + random_file)["obs"]
  encoding = np.load(ENCODING_DATA_DIR + random_file)
  mu = encoding["mu"]
  sigma = encoding["sigma"]
  action = encoding["action"]
  
  # Sample z from mu and sigma.
  z = mu + sigma * np.random.randn(MAX_FRAME, Z_LENGTH)
  za = np.reshape(np.concatenate((z[:-1, :], action[:-1, :]), -1), (1, MAX_FRAME - 1, Z_LENGTH + A_LENGTH))
  
  # Load RNN-MDN model.
  rnn = RNN_MDN()
  rnn.build_model(is_training = False, is_assigning = False, is_single_input = False)
  
  with tf.Session(graph = rnn.graph) as sess:
    # Load variables.
    saver = tf.train.Saver()
    saver.restore(sess, SAVE_RNN_DIR + rnn_file_name)
    
    # Compute the key parameters.
    logits, mu, sigma = sess.run([rnn.logits, rnn.mu, rnn.sigma], feed_dict = {rnn.ZA: za})
    length = len(logits)
    
    # Sample next_z from logits, mu and sigma.
    reduced_logits = logits - np.max(logits, -1, keepdims = True)
    pi = np.exp(reduced_logits / temperature) / np.sum(np.exp(reduced_logits / temperature), -1, keepdims = True)
    chosen_mode = np.reshape(np.array([np.random.choice(MODES, p = x) for x in pi]), [-1, 1])
    chosen_mu = np.reshape(np.array([mu[i, chosen_mode[i]] for i in range(length)]), [-1, 1])
    chosen_sigma = np.reshape(np.array([sigma[i, chosen_mode[i]] for i in range(length)]), [-1, 1])
    next_z = chosen_mu + chosen_sigma * np.random.randn(length, 1) * np.sqrt(temperature)
    next_z = np.reshape(next_z, (MAX_FRAME - 1, Z_LENGTH))
    
    # Add z[0] to next_z.
    next_z = np.concatenate((np.reshape(z[0], (1, -1)), next_z), 0)
  tf.contrib.keras.backend.clear_session()
  
  # Load VAE model.
  vae = VAE()
  vae.build_model(is_training = False, is_assigning = False)
  
  with tf.Session(graph = vae.graph) as sess:
    # Load variables.
    saver = tf.train.Saver()
    saver.restore(sess, SAVE_VAE_DIR + vae_file_name)
    
    # Compute the reconstruction from direct encoding.
    recons_from_z = sess.run(vae.output, feed_dict = {vae.z: z})
    # Compute the reconstruction from predicted encoding.
    recons_from_next_z = sess.run(vae.output, feed_dict = {vae.z: next_z})
  tf.contrib.keras.backend.clear_session()
  
  imageio.mimsave(FIGURE_RNN_VISUALIZATION_DIR + random_file_name + ".gif", [plot_obs_recons(obs[i], recons_from_z[i], recons_from_next_z[i]) for i in range(MAX_FRAME)], fps = 20)
  
def plot_obs_recons(obs, recons_from_z, recons_from_next_z):
  # Plot the observation and reconstruction.
  f, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (6, 2.1))
  ax[0].imshow(obs)
  ax[0].set_title("Observation")
  ax[0].set_axis_off()
  ax[1].imshow(recons_from_z)
  ax[1].set_title("Reconstruction")
  ax[1].set_axis_off()
  ax[2].imshow(recons_from_next_z)
  ax[2].set_title("Prediction")
  ax[2].set_axis_off()
  f.tight_layout()
  
  f.canvas.draw()
  image = np.frombuffer(f.canvas.tostring_rgb(), dtype = "uint8")
  image = image.reshape(f.canvas.get_width_height()[::-1] + (3,))
  plt.close(f)

  return image

if __name__ == "__main__":
  # e.g. rnn_visualization(random_file = "0945.npz")
  rnn_visualization()