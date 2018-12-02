import gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

import config
from policy import Policy
from wrappers import make_atari

# Local hyperparameters.
ENV_NAME = config.ENV_NAME
AUXILIARY_TASK = config.AUXILIARY_TASK

SAVE_DIR = config.SAVE_DIR
FIGURE_DIR = config.FIGURE_DIR

def test():
  '''
  This function visualizes the game play. The environment will be reset immediately and the game will not be recorded.
  To record the game play, please run the record() function.
  '''
  save_path = SAVE_DIR + ENV_NAME + "/" + AUXILIARY_TASK + "/"
  
  obs_mean_std = np.load(save_path + "obs_mean_std.npz")
  obs_mean = obs_mean_std["obs_mean"]
  obs_std = obs_mean_std["obs_std"]
  
  # Create environment.
  env = make_atari(ENV_NAME)
  obs_space = env.observation_space
  action_space = env.action_space
  
  # Build models.
  policy = Policy(obs_space, action_space, is_training = False)
  
  with tf.Session() as sess:
    # Load variables.
    saver_policy = tf.train.Saver(policy.trainable_variables)
    saver_policy.restore(sess, save_path + "policy")
    
    total_step = 0
    total_reward = 0
    while True:
      # Get observation.
      if total_step == 0:
        obs = env.reset()
      else:
        obs = obs_next
      obs = (obs - obs_mean) / obs_std
      env.render()
      # Get action.
      action = sess.run(policy.action, feed_dict = {policy.Obs: np.reshape(obs, [1, 1, *obs.shape])})
      action = np.squeeze(action, (0, 1))
      
      # Interact with the environment.
      obs_next, reward, done, _ = env.step(action)
      total_reward += reward
      if done:
        # Reset environment.
        print("Episodic reward: ", total_reward, sep = "")
        obs_next = env.reset()
        total_reward = 0
      # Update step counter.
      total_step += 1
    env.close()

def record():
  '''
  This function generates a gif file for a single episode. This process may take some time.
  To watch the non-stop game play, please run the test() function.
  '''
  save_path = SAVE_DIR + ENV_NAME + "/" + AUXILIARY_TASK + "/"
  figure_path = FIGURE_DIR + ENV_NAME + "/" + AUXILIARY_TASK + "/"
  
  list_obs = []
  list_reward = []
  
  obs_mean_std = np.load(save_path + "obs_mean_std.npz")
  obs_mean = obs_mean_std["obs_mean"]
  obs_std = obs_mean_std["obs_std"]
  
  # Create environment.
  env = make_atari(ENV_NAME)
  obs_space = env.observation_space
  action_space = env.action_space
  
  # Build models.
  policy = Policy(obs_space, action_space, is_training = False)
  
  with tf.Session() as sess:
    # Load variables.
    saver_policy = tf.train.Saver(policy.trainable_variables)
    saver_policy.restore(sess, save_path + "policy")
    
    total_reward = 0
    obs = env.reset()
    while True:
      list_obs.append(obs)
      list_reward.append(total_reward)
      env.render()
      # Get observation.
      obs = (obs - obs_mean) / obs_std
      # Get action.
      action = sess.run(policy.action, feed_dict = {policy.Obs: np.reshape(obs, [1, 1, *obs.shape])})
      action = np.squeeze(action, (0, 1))
      
      # Interact with the environment.
      obs, reward, done, _ = env.step(action)
      total_reward += reward
      if done:
        list_obs.append(obs)
        list_reward.append(total_reward)
        break
  env.close()
  
  # Record the gameplay.
  imageio.mimsave(figure_path + "gameplay.gif", [plot_obs(obs, reward) for obs, reward in zip(list_obs, list_reward)], fps = 30)
  
def plot_obs(obs, reward):
  # Plot the observation.
  f, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (2, 2))
  ax.imshow(obs)
  ax.set_title("Reward: " + format(int(reward), "04d"))
  ax.set_axis_off()
  
  f.canvas.draw()
  image = np.frombuffer(f.canvas.tostring_rgb(), dtype = "uint8")
  image = image.reshape(f.canvas.get_width_height()[::-1] + (3,))
  plt.close(f)
  return image

if __name__ == "__main__":
  record()