import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

import config
from controller import Controller
from env import MyCarRacing
from rnn_mdn import RNN_MDN
from vae import VAE

# Local hyperparameters
CONTROLLER_MODE = config.CONTROLLER_MODE
MAX_FRAME = config.MAX_FRAME
RENDER_MODE = config.RENDER_MODE

SAVE_VAE_DIR = config.SAVE_VAE_DIR
SAVE_RNN_DIR = config.SAVE_RNN_DIR
SAVE_CONTROLLER_DIR = config.SAVE_CONTROLLER_DIR
FIGURE_CONTROLLER_VISUALIZATION_DIR = config.FIGURE_CONTROLLER_VISUALIZATION_DIR

def controller_visualization(vae_file_name = "vae", rnn_file_name = "rnn", con_file_name = "controller"):
  # Create folders.
  if not os.path.isdir(FIGURE_CONTROLLER_VISUALIZATION_DIR):
    os.makedirs(FIGURE_CONTROLLER_VISUALIZATION_DIR)
  
  # Load models.
  vae = VAE()
  vae.build_model(is_training = False, is_assigning = True)
  rnn = RNN_MDN()
  rnn.build_model(is_training = False, is_assigning = True, is_single_input = True)
  con = Controller()
  
  sess_vae = tf.Session(graph = vae.graph)
  sess_rnn = tf.Session(graph = rnn.graph)
  
  # Load variables.
  with vae.graph.as_default():
    saver_vae = tf.train.Saver()
    saver_vae.restore(sess_vae, SAVE_VAE_DIR + vae_file_name)
  with rnn.graph.as_default():
    saver_rnn = tf.train.Saver()
    saver_rnn.restore(sess_rnn, SAVE_RNN_DIR + rnn_file_name)
  con_vars = np.load(SAVE_CONTROLLER_DIR + con_file_name + ".npz")
  con.weights = con_vars["weights"]
  con.bias = con_vars["bias"]
  
  list_obs = []
  list_reward = []
  total_reward = 0
  env = MyCarRacing()
  
  # Get intitial state of RNN.
  state = sess_rnn.run(rnn.initial_state)
  
  # Reset the environment.
  obs = env.reset()
  
  for step in range(MAX_FRAME):
    env.render(mode = RENDER_MODE)
    list_obs.append(obs)
    list_reward.append(total_reward)
    
    # Encode the observation.
    obs = np.reshape(obs / 255.0, (1, 64, 64, 3))
    z, recons = sess_vae.run([vae.z, vae.output], feed_dict = {vae.Input: obs})
    
    # Get action.
    if CONTROLLER_MODE == "Z":
      controller_input = z
    elif CONTROLLER_MODE == "ZH":
      controller_input = np.concatenate((z, state.h), 1)
    else:
      controller_input = np.concatenate((z, state.h, state.c), 1)
    action = con.get_action(controller_input)
    
    # Update the hidden state.
    za = np.reshape(np.concatenate((z, action), 1), (1, 1, -1))
    state = sess_rnn.run(rnn.final_state, feed_dict = {rnn.ZA: za, rnn.initial_state: state})
    
    # Interact with the game engine.
    obs, reward, done, _ = env.step(action[0])
    total_reward += reward
    
    if done:
      break
  
  print(total_reward)
  env.render(close = True)
  sess_vae.close()
  sess_rnn.close()
  tf.contrib.keras.backend.clear_session()
  
  index = len(os.listdir(FIGURE_CONTROLLER_VISUALIZATION_DIR))
  imageio.mimsave(FIGURE_CONTROLLER_VISUALIZATION_DIR + format(index, "04d") + ".gif", [plot_obs(list_obs[i], list_reward[i]) for i in range(len(list_obs))], fps = 20)

def plot_obs(obs, reward):
  # Plot the observation.
  f, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (2, 2))
  ax.imshow(obs)
  ax.set_title("Reward: " + format(reward, "3.1f"))
  ax.set_axis_off()
  
  f.canvas.draw()
  image = np.frombuffer(f.canvas.tostring_rgb(), dtype = "uint8")
  image = image.reshape(f.canvas.get_width_height()[::-1] + (3,))
  plt.close(f)

  return image

if __name__ == "__main__":
  controller_visualization()