import multiprocessing
import numpy as np
import os

import config
from reacher import MyReacherEnv

# Local hyperparameters
A_LENGTH = config.A_LENGTH
ACTION_SCALING = config.ACTION_SCALING
MAX_FRAME = config.MAX_FRAME
EXPLORATION_ACTION_NOISE = 0.5
MAX_EXPLORATION_ACTION_NOISE = 1.0
MAX_EPISODE = 200
PROCESS = 1

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
    
    # Reset the environment.
    env.reset()
    
    for step in range(MAX_FRAME):
      obs = env.render(mode = "rgb_array")
      list_obs.append(obs)
      
      # Sample random action.
      noise = np.clip(EXPLORATION_ACTION_NOISE * np.random.randn(A_LENGTH), -MAX_EXPLORATION_ACTION_NOISE, MAX_EXPLORATION_ACTION_NOISE)
      a = np.clip(ACTION_SCALING * noise, -ACTION_SCALING, ACTION_SCALING)
      
      # Interact with the game engine.
      env.step(a)
    
    # Save file.
    list_obs = np.array(list_obs, dtype = np.uint8)
    np.savez_compressed(RAW_DATA_DIR + format(start_index + episode, "04d"), obs = list_obs)
  
  env.close()

if __name__ == "__main__":
  random_sampling()