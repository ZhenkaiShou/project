import cv2
import gym
import imageio
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from atari_wrappers import make_atari
from config import *
from model import QValueNetwork

def visualize_attention(file_name):
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
  list_gradients_exp = []
  list_gradients_adv = []
  
  with tf.Session() as sess:
    # Load network parameters.
    saver = tf.train.Saver(var_list = main_network.variables)
    saver.restore(sess, SAVE_DIR + file_name)
    
    done = False
    while True:
      # Get the raw observation.
      raw_obs = env.render(mode = "rgb_array")
      list_obs.append(raw_obs)
      
      # Compute gradients w.r.t. the raw observation.
      gradients_exp, gradients_adv = sess.run([main_network.obs_gradients_exp, main_network.obs_gradients_adv], feed_dict = {main_network.Obs: np.expand_dims(np.array(obs) / 255.0, 0)})
      gradients_exp = gradients_exp[0]
      gradients_adv = gradients_adv[0]
      list_gradients_exp.append(gradients_exp)
      list_gradients_adv.append(gradients_adv)
      
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
        # Compute the last observation gradients.
        gradients_exp, gradients_adv = sess.run([main_network.obs_gradients_exp, main_network.obs_gradients_adv], feed_dict = {main_network.Obs: np.expand_dims(np.array(obs) / 255.0, 0)})
        gradients_exp = gradients_exp[0]
        gradients_adv = gradients_adv[0]
        list_gradients_exp.append(gradients_exp)
        list_gradients_adv.append(gradients_adv)
        break
      # Update the observation.
      obs = obs_next
  
  env.close()
  
  # Record the gameplay.
  imageio.mimsave(FIGURE_VISUALIZATION_DIR + "attention.gif", 
    [plot_attention(obs, gradients_exp, gradients_adv) for obs, gradients_exp, gradients_adv in zip(list_obs, list_gradients_exp, list_gradients_adv)], fps = 30)

def plot_attention(obs, gradients_exp, gradients_adv, scale = 0.01):
  # Transform graidents into RGBD heatmaps.
  attention_exp = np.zeros((110, 84, 4))
  attention_adv = np.zeros((110, 84, 4))
  attention_exp[:, :, 0] = 1
  attention_adv[:, :, 0] = 1
  attention_exp[18:102, :, 3] = np.clip(8 * np.abs(gradients_exp), 0, 1)
  attention_adv[18:102, :, 3] = np.clip(8 * np.abs(gradients_adv), 0, 1)
  target_size = (np.shape(obs)[1], np.shape(obs)[0])
  attention_exp = cv2.resize(attention_exp, target_size, interpolation=cv2.INTER_AREA)
  attention_adv = cv2.resize(attention_adv, target_size, interpolation=cv2.INTER_AREA)
  
  f, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (2 * np.shape(obs)[1] * scale, 1.1 * np.shape(obs)[0] * scale))
  # Plot the attention w.r.t. expectation.
  ax[0].imshow(obs)
  ax[0].imshow(attention_exp)
  ax[0].set_axis_off()
  ax[0].set_title("Expectation")
  
  # Plot the attention w.r.t. advantage.
  ax[1].imshow(obs)
  ax[1].imshow(attention_adv)
  ax[1].set_axis_off()
  ax[1].set_title("Advantage")
  
  f.canvas.draw()
  image = np.frombuffer(f.canvas.tostring_rgb(), dtype = "uint8")
  image = image.reshape(f.canvas.get_width_height()[::-1] + (3,))
  plt.close(f)
  return image

if __name__ == "__main__":
  visualize_attention(file_name = "dqn")