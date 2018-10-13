import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

import config
from vae import VAE

# Local hyperparameters
MAX_FRAME = config.MAX_FRAME

SAVE_VAE_DIR = config.SAVE_VAE_DIR
RAW_DATA_DIR = config.RAW_DATA_DIR
FIGURE_VAE_VISUALIZATION_DIR = config.FIGURE_VAE_VISUALIZATION_DIR

def vae_visualization(file_name = "vae", random_file = None):
  # Create folders.
  if not os.path.isdir(FIGURE_VAE_VISUALIZATION_DIR):
    os.makedirs(FIGURE_VAE_VISUALIZATION_DIR)
  
  if random_file == None:
    # Load random data.
    file_list = os.listdir(RAW_DATA_DIR)
    random_file = np.random.choice(file_list)
  random_file_name = os.path.splitext(random_file)[0]
  obs = np.load(RAW_DATA_DIR + random_file)["obs"]
  
  # Load models.
  vae = VAE(name = "vae")
  
  with tf.Session() as sess:
    # Load variables.
    saver = tf.train.Saver()
    saver.restore(sess, SAVE_VAE_DIR + file_name)
    
    # Compute the reconstruction.
    z = sess.run(vae.mu, feed_dict = {vae.Input: obs / 255.0})
    recons = sess.run(vae.output, feed_dict = {vae.z: z})
  
  tf.contrib.keras.backend.clear_session()
  
  imageio.mimsave(FIGURE_VAE_VISUALIZATION_DIR + random_file_name + ".gif", [plot_obs_recons(obs[i], recons[i]) for i in range(MAX_FRAME)], fps = 20)
  
def plot_obs_recons(obs, recons):
  # Plot the observation and reconstruction.
  f, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (4, 2.1))
  ax[0].imshow(obs)
  ax[0].set_title("Observation")
  ax[0].set_axis_off()
  ax[1].imshow(recons)
  ax[1].set_title("Reconstruction")
  ax[1].set_axis_off()
  f.tight_layout()
  
  f.canvas.draw()
  image = np.frombuffer(f.canvas.tostring_rgb(), dtype = "uint8")
  image = image.reshape(f.canvas.get_width_height()[::-1] + (3,))
  plt.close(f)

  return image

if __name__ == "__main__":
  # e.g. vae_visualization(random_file = "0945.npz")
  vae_visualization()