import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

import config
from rnn import RNN
from vae import VAE

# Local hyperparameters
MAX_FRAME = config.MAX_FRAME

SAVE_VAE_DIR = config.SAVE_VAE_DIR
SAVE_RNN_DIR = config.SAVE_RNN_DIR
RAW_DATA_DIR = config.RAW_DATA_DIR
ENCODING_DATA_DIR = config.ENCODING_DATA_DIR
FIGURE_RNN_VISUALIZATION_DIR = config.FIGURE_RNN_VISUALIZATION_DIR

def rnn_visualization(vae_file_name = "vae", rnn_file_name = "rnn", random_file = None):
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
  z = encoding["z"]
  a = encoding["action"]
  
  # Load VAE and RNN models.
  vae = VAE(name = "vae")
  rnn = RNN(name = "rnn", is_single_input = False)
  
  with tf.Session() as sess:
    # Load variables.
    saver_vae = tf.train.Saver(vae.trainable_variables)
    saver_vae.restore(sess, SAVE_VAE_DIR + vae_file_name)
    saver_rnn = tf.train.Saver(rnn.trainable_variables)
    saver_rnn.restore(sess, SAVE_RNN_DIR + rnn_file_name)
    
    # Compute z_next.
    z_next = sess.run(rnn.z_next, feed_dict = {rnn.Z: np.reshape(z[:-1], (1, *z[:-1].shape)), rnn.A: np.reshape(a[:-1], (1, *a[:-1].shape))})
    z_next = z_next[0]
    # Add z[0] to z_next.
    z0 = np.reshape(z[0], (1, *z[0].shape))
    z_next = np.concatenate((z0, z_next), 0)
    
    # Compute the reconstruction from direct encoding.
    recons_from_z = sess.run(vae.output, feed_dict = {vae.z: z})
    # Compute the reconstruction from predicted encoding.
    recons_from_z_next = sess.run(vae.output, feed_dict = {vae.z: z_next})
  tf.contrib.keras.backend.clear_session()
  
  imageio.mimsave(FIGURE_RNN_VISUALIZATION_DIR + random_file_name + ".gif", [plot_obs_recons(obs[i], recons_from_z[i], recons_from_z_next[i]) for i in range(MAX_FRAME)], fps = 20)
  
def plot_obs_recons(obs, recons_from_z, recons_from_z_next):
  # Plot the observation and reconstruction.
  f, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (6, 2.1))
  ax[0].imshow(obs)
  ax[0].set_title("Observation")
  ax[0].set_axis_off()
  ax[1].imshow(recons_from_z)
  ax[1].set_title("Reconstruction")
  ax[1].set_axis_off()
  ax[2].imshow(recons_from_z_next)
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