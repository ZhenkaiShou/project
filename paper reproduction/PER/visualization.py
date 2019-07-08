import gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from atari_wrappers import make_atari
from config import *
from model_graph import ModelGraph

def visualize(env_name, file_name, network_type):
  # Create folders.
  if not os.path.isdir(FIGURE_VISUALIZATION_DIR):
    os.makedirs(FIGURE_VISUALIZATION_DIR)
  
  # Obtain environment parameters.
  env = make_atari(env_name)
  obs_shape = env.observation_space.shape
  num_action = env.action_space.n
  
  # Build model graph.
  model_graph = ModelGraph(obs_shape, num_action, network_type = network_type)
  
  # Initialize session and load variables.
  sess = tf.InteractiveSession()
  model_graph.load(SAVE_DIR + file_name)
  
  obs = env.reset()
  list_obs = []
  
  while True:
    # Get the raw observation.
    raw_obs = env.render(mode = "rgb_array")
    list_obs.append(raw_obs)
    
    env.render()
    # Get action.
    action = model_graph.act(np.expand_dims(np.array(obs), 0))
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
  f, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (np.shape(obs)[1] * scale, np.shape(obs)[0] * scale))
  ax.imshow(obs)
  ax.set_axis_off()
  
  f.canvas.draw()
  image = np.frombuffer(f.canvas.tostring_rgb(), dtype = "uint8")
  image = image.reshape(f.canvas.get_width_height()[::-1] + (3,))
  plt.close(f)
  return image

if __name__ == "__main__":
  visualize(env_name = "SeaquestNoFrameskip-v4", file_name = "seaquest_prioritized", network_type = "conv")