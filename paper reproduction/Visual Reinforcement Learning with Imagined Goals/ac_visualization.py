import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

import config
from ac import Actor, Critic
from reacher import MyReacherEnv
from rnn import RNN
from vae import VAE

# Local hyperparameters
Z_LENGTH = config.Z_LENGTH
A_LENGTH = config.A_LENGTH
ACTION_SCALING = config.ACTION_SCALING
REWARD_SCALING = config.REWARD_SCALING
MAX_FRAME = config.MAX_FRAME

SAVE_VAE_DIR = config.SAVE_VAE_DIR
SAVE_RNN_DIR = config.SAVE_RNN_DIR
SAVE_AC_DIR = config.SAVE_AC_DIR
FIGURE_AC_VISUALIZATION_DIR = config.FIGURE_AC_VISUALIZATION_DIR

def ac_training(save_file_name = "ac", vae_file_name = "vae", rnn_file_name = "rnn", ac_file_name = "ac"):
  # Create folders.
  if not os.path.isdir(FIGURE_AC_VISUALIZATION_DIR):
    os.makedirs(FIGURE_AC_VISUALIZATION_DIR)
  
  # Load models.
  vae = VAE(name = "vae")
  rnn = RNN(name = "rnn", is_single_input = True)
  actor = Actor(name = "actor")
  critic = Critic(name = "critic")
  
  # Initialization.
  env = MyReacherEnv()
  
  with tf.Session() as sess:
    # Load network parameters.    
    saver_vae = tf.train.Saver(vae.trainable_variables)
    saver_rnn = tf.train.Saver(rnn.trainable_variables)
    saver_ac = tf.train.Saver([*actor.trainable_variables, *critic.trainable_variables])
    saver_vae.restore(sess, SAVE_VAE_DIR + vae_file_name)
    saver_rnn.restore(sess, SAVE_RNN_DIR + rnn_file_name)
    saver_ac.restore(sess, SAVE_AC_DIR + ac_file_name)
    
    list_obs = []
    list_distance = []
    
    # Get intitial state of RNN.
    state = sess.run(rnn.initial_state)
    
    # Sample goals from real game.
    env.reset()
    goal = env.render(mode = "rgb_array")
    z_goal = sess.run(vae.mu, feed_dict = {vae.Input: np.reshape(goal / 255.0, (1, *goal.shape))})
    
    # Reset the environment.
    env.reset()
    
    # Get the first observation.
    obs = env.render(mode = "rgb_array")
    while np.linalg.norm(obs - goal) == 0:
      obs = env.render(mode = "rgb_array")
    # Encode the observation.
    z = sess.run(vae.mu, feed_dict = {vae.Input: np.reshape(obs / 255.0, (1, *obs.shape))})
    
    for step in range(MAX_FRAME):
      # Get the action.
      a = sess.run(actor.pi, feed_dict = {actor.Z: z, actor.H: np.concatenate([state.h, state.c], 1), actor.ZGoal: z_goal})
      
      # Update the hidden state.
      state_next = sess.run(rnn.final_state, feed_dict = {rnn.Z: np.reshape(z, (1, *z.shape)), rnn.A: np.reshape(a, (1, *a.shape)), rnn.initial_state: state})
      
      # Interact with the game engine.
      env.step(a[0])
      
      # Get the next observation.
      obs_next = env.render(mode = "rgb_array")
      list_obs.append(obs_next)
      # Encode the next observation.
      z_next = sess.run(vae.mu, feed_dict = {vae.Input: np.reshape(obs_next / 255.0, (1, *obs_next.shape))})
      # Compute distance.
      distance = REWARD_SCALING * np.linalg.norm(z_next - z_goal, axis = 1)[0]
      list_distance.append(distance)
      
      obs = obs_next
      z = z_next
      state = state_next
    
  tf.contrib.keras.backend.clear_session()
  env.close()
  
  imageio.mimsave(FIGURE_AC_VISUALIZATION_DIR + save_file_name + ".gif", [plot_episode(goal, list_obs[i], list_distance[i]) for i in range(MAX_FRAME)], fps = 20)
  
def plot_episode(goal, obs, distance):
  # Plot the goal and observation.
  f, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (4, 2.1))
  ax[0].imshow(goal)
  ax[0].set_title("Goal")
  ax[0].set_axis_off()
  ax[1].imshow(obs)
  ax[1].set_title("d = " + format(distance, ".8f"))
  ax[1].set_axis_off()
  f.tight_layout()
  
  f.canvas.draw()
  image = np.frombuffer(f.canvas.tostring_rgb(), dtype = "uint8")
  image = image.reshape(f.canvas.get_width_height()[::-1] + (3,))
  plt.close(f)

  return image

if __name__ == "__main__":
  ac_training()