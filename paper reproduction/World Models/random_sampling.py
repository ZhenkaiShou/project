import multiprocessing
import numpy as np
import os

import config
from controller import Controller
from env import MyCarRacing
from rnn_mdn import RNN_MDN
from vae import VAE

# Local hyperparameters
MAX_FRAME = config.MAX_FRAME
MAX_EPISODE = config.MAX_EPISODE
PROCESS = 2
CONTROLLER_MODE = config.CONTROLLER_MODE
RENDER_MODE = config.RENDER_MODE

RAW_DATA_DIR = config.RAW_DATA_DIR

def random_sampling():
  # Create folder.
  if not os.path.isdir(RAW_DATA_DIR):
    os.makedirs(RAW_DATA_DIR)
  
  # Determine how many episodes to run for each process.
  start_index = 0
  base = MAX_EPISODE // PROCESS
  reminder = MAX_EPISODE % PROCESS
  arg_list = [None for _ in range(PROCESS)]
  for i in range(PROCESS):
    max_episode = base + 1 if i < reminder else base
    arg_list[i] = [start_index, max_episode]
    start_index += max_episode
  
  with multiprocessing.Pool(PROCESS) as p:
    p.starmap(random_sampling_process, arg_list)

def random_sampling_process(start_index, max_episode):
  import tensorflow as tf
  
  np.random.seed()
  
  # Load models.
  vae = VAE()
  vae.build_model(is_training = False, is_assigning = True)
  rnn = RNN_MDN()
  rnn.build_model(is_training = False, is_assigning = True, is_single_input = True)
  con = Controller()
  
  sess_vae = tf.Session(graph = vae.graph)
  sess_rnn = tf.Session(graph = rnn.graph)
  
  env = MyCarRacing()
  
  for episode in range(max_episode):
    # Initialize the networks with random parameters.
    stddev = 0.01 * np.random.rand()
    
    with vae.graph.as_default():
      for i in range(len(tf.trainable_variables())):
        random = stddev * np.random.standard_cauchy(tf.trainable_variables()[i].get_shape()) / 10000
        sess_vae.run(vae.assign_op[i], feed_dict = {vae.Assigned_Value: random})
    with rnn.graph.as_default():
      for i in range(len(tf.trainable_variables())):
        random = stddev * np.random.standard_cauchy(tf.trainable_variables()[i].get_shape()) / 10000
        sess_rnn.run(rnn.assign_op[i], feed_dict = {rnn.Assigned_Value: random})
    random = stddev * np.random.standard_cauchy(np.prod(np.shape(con.weights)) + np.prod(np.shape(con.bias)))
    con.weights = np.reshape(random[:np.prod(np.shape(con.weights))], np.shape(con.weights))
    con.bias = np.reshape(random[-np.prod(np.shape(con.bias)):], np.shape(con.bias))
    
    # Get intitial state of RNN.
    state = sess_rnn.run(rnn.initial_state)
    
    list_obs = []
    list_action = []
    
    # Reset the environment.
    obs = env.reset()
    
    for step in range(MAX_FRAME):
      env.render(mode = RENDER_MODE)
      list_obs.append(obs)
      
      # Encode the observation.
      obs = np.reshape(obs / 255.0, (1, 64, 64, 3))
      z = sess_vae.run(vae.z, feed_dict = {vae.Input: obs})
      
      # Get action.
      if CONTROLLER_MODE == "Z":
        controller_input = z
      elif CONTROLLER_MODE == "ZH":
        controller_input = np.concatenate((z, state.h), 1)
      else:
        controller_input = np.concatenate((z, state.h, state.c), 1)
      action = con.get_action(controller_input)
      list_action.append(action[0])
      
      # Update the hidden state.
      za = np.reshape(np.concatenate((z, action), 1), (1, 1, -1))
      state = sess_rnn.run(rnn.final_state, feed_dict = {rnn.ZA: za, rnn.initial_state: state})
      
      # Interact with the game engine.
      obs, reward, done, _ = env.step(action[0])
    
    # Save file.
    list_obs = np.array(list_obs, dtype = np.uint8)
    list_action = np.array(list_action, dtype = np.float16)
    np.savez_compressed(RAW_DATA_DIR + format(start_index + episode, "04d"), obs = list_obs, action = list_action)
  
  env.render(close = True)
  sess_vae.close()
  sess_rnn.close()
  tf.contrib.keras.backend.clear_session()

if __name__ == "__main__":
  random_sampling()