import multiprocessing
import numpy as np
import os

import config
from reacher import MyReacherEnv

# Local hyperparameters
A_LENGTH = config.A_LENGTH
ACTION_SCALING = config.ACTION_SCALING
MAX_FRAME = config.MAX_FRAME
ACTION_NOISE = 0.5
MAX_EPISODE = 5000
PROCESS = 2

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
  np.random.seed()
  env = MyReacherEnv()
  
  for episode in range(max_episode):
    list_obs = []
    list_action = []
    
    # Reset the environment.
    env.reset()
    
    for step in range(MAX_FRAME):
      obs = env.render(mode = "rgb_array")
      list_obs.append(obs)
      
      # Sample random action.
      noise = ACTION_NOISE * np.random.randn(A_LENGTH)
      action = np.clip(ACTION_SCALING * noise, -ACTION_SCALING, ACTION_SCALING)
      list_action.append(action)
      
      # Interact with the game engine.
      env.step(action)
    
    # Save file.
    list_obs = np.array(list_obs, dtype = np.uint8)
    list_action = np.array(list_action, dtype = np.float16)
    np.savez_compressed(RAW_DATA_DIR + format(start_index + episode, "04d"), obs = list_obs, action = list_action)
  
  env.close()

if __name__ == "__main__":
  random_sampling()