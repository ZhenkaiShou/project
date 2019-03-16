import gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from atari_wrappers import make_atari
from config import *
from model import QValueNetwork

def visualize(file_name):
  # Create folders.
  if not os.path.isdir(FIGURE_VISUALIZATION_DIR):
    os.makedirs(FIGURE_VISUALIZATION_DIR)
  
  # Obtain environment parameters.
  env = make_atari(ENV_NAME)
  obs_space = env.observation_space
  action_space = env.action_space
  
  # Only build main network for visualization.
  main_network = QValueNetwork(obs_space, action_space, name = "main_network")
  
  obs = env.reset()
  list_obs = []
  
  with tf.Session() as sess:
    # Load network parameters.
    saver = tf.train.Saver(var_list = main_network.variables)
    saver.restore(sess, SAVE_DIR + file_name)
    
    done = False
    while True:
      # Get the raw observation.
      raw_obs = env.render(mode = "rgb_array")
      list_obs.append(raw_obs)
      
      env.render()
      # Get action.
      q = sess.run(main_network.q, feed_dict = {main_network.Obs: np.expand_dims(np.array(obs) / 255.0, 0)})
      action = np.argmax(q[0])
      # Interact with the environment.
      obs_next, reward, done, _ = env.step(action)
      if done:
        # Get the last raw observation.
        raw_obs = env.render(mode = "rgb_array")
        list_obs.append(raw_obs)
        break
      # Update the observation.
      obs = obs_next
  
  env.close()
  
  # Record the gameplay.
  imageio.mimsave(FIGURE_VISUALIZATION_DIR + "gameplay.gif", [plot_obs(obs) for obs in list_obs], fps = 30)

def plot_obs(obs, scale = 0.01):
  # Plot the observation.
  f, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (np.shape(obs)[0] * scale, np.shape(obs)[1] * scale))
  ax.imshow(obs)
  ax.set_axis_off()
  
  f.canvas.draw()
  image = np.frombuffer(f.canvas.tostring_rgb(), dtype = "uint8")
  image = image.reshape(f.canvas.get_width_height()[::-1] + (3,))
  plt.close(f)
  return image

if __name__ == "__main__":
  visualize(file_name = "par_dqn")