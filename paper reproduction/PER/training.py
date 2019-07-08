import gym
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import time

from atari_wrappers import make_atari
from model_graph import ModelGraph
from config import *
from plot import plot
from replay_buffer import PrioritizedReplayBuffer
from schedules import LinearSchedule, StaircaseSchedule

def train(
  env_name,
  file_name, 
  network_type,
  env_seed = None,
  seed = None,
  
  buffer_size = int(1e5),
  alpha = 0.6,
  batch_size = 32,
  reward_min = -1.0,
  reward_max = 1.0,
  reward_discount = 0.99,
  
  epsilon_start = 1.0,
  epsilon_end = 0.05,
  epsilon_decay_step = int(1e6),
  beta_start = 0.4,
  beta_end = 1.0,
  beta_decay_step = int(1e7),
  lrs = [5e-5, 5e-6],
  lr_cutoff_steps = [int(8e6)],
  
  total_steps = int(1e7),
  initial_buffer_size = int(1e5),
  target_network_update_step = 1000,
  training_step = 4,
  last_k_episodes = 100,
  print_frequency = 10,
  save_frequency = 0.01
  ):
  # Create folders.
  if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR)
  if not os.path.isdir(CSV_DIR):
    os.makedirs(CSV_DIR)
  
  # Create environment.
  env = make_atari(env_name)
  if env_seed is not None:
    env.seed(env_seed)
  obs_shape = env.observation_space.shape
  num_action = env.action_space.n
  
  if seed is not None:
    np.random.seed(seed)
    tf.set_random_seed(seed)
  
  # Initialize step schedules.
  epsilon = LinearSchedule(start = epsilon_start, end = epsilon_end, decay_step = epsilon_decay_step)
  beta = LinearSchedule(start = epsilon_start, end = epsilon_end, decay_step = epsilon_decay_step)
  learning_rate = StaircaseSchedule(values = lrs, cutoff_steps = lr_cutoff_steps)
  
  # Initialize replay buffer.
  replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha = alpha)
  
  # Build model graph.
  model_graph = ModelGraph(obs_shape, num_action, network_type = network_type, gamma = reward_discount)
  
  # Initialize session and variables.
  sess = tf.InteractiveSession()
  model_graph.initialize_variables()
  model_graph.update_target_network()
  
  start_time = time.time()
  list_step = []
  list_episodic_reward = []
  list_mean_episodic_reward = []
  episodic_reward = 0
  highest_episodic_reward = None
  
  obs = env.reset()
  for step in range(1, total_steps):
    # Synchronize the target network periodically (target network <- main network).
    if step > initial_buffer_size and step % target_network_update_step == 0:
      model_graph.update_target_network()
    
    # Sample action with epsilon-greedy policy.
    action = model_graph.epsilon_act(np.expand_dims(obs, axis = 0), epsilon.get_value(step))[0]
    
    # Interact with the environment.
    obs_next, reward, done, _ = env.step(action)
    episodic_reward += reward
    if done:
      obs_next = env.reset()
      
      # Record episodic reward.
      list_step.append(step)
      list_episodic_reward.append(episodic_reward)
      mean_episodic_reward = np.round(np.mean(list_episodic_reward[-last_k_episodes:]), 2)
      list_mean_episodic_reward.append(mean_episodic_reward)
      if len(list_episodic_reward) % print_frequency == 0:
        print("Episode ", str(len(list_episodic_reward)), ": step = ", step, ", mean reward = ", mean_episodic_reward, ".", sep = "")
      
      # Save the network when the mean episodic reward breaks the record.
      if step >= initial_buffer_size and len(list_episodic_reward) >= last_k_episodes:
        if highest_episodic_reward is None or mean_episodic_reward > highest_episodic_reward:
          if np.random.uniform() < save_frequency:
            model_graph.save(SAVE_DIR + file_name)
            print("Save the network as mean episodic reward increases from ", highest_episodic_reward, " to ", mean_episodic_reward, ".", sep = "")
            highest_episodic_reward = mean_episodic_reward
      
      episodic_reward = 0
    # Store data.
    data = (obs, action, reward, done, obs_next)
    replay_buffer.append(data)
    # Update observation.
    obs = obs_next
    
    # Train the agent.
    if step > initial_buffer_size and step % training_step == 0:
      # Sample training data from the replay buffer.
      batch_index, batch_data, batch_weights = replay_buffer.sample(batch_size, beta.get_value(step))
      batch_obs, batch_action, batch_reward, batch_done, batch_obs_next = \
        [np.array([batch_data[j][i] for j in range(batch_size)]) for i in range(len(batch_data[0]))]
      
      # Clip the reward.
      batch_reward = np.clip(batch_reward, reward_min, reward_max)
      
      # One train step.
      td_error = model_graph.train(batch_obs, batch_action, batch_reward, batch_done, batch_obs_next, batch_weights, learning_rate.get_value(step))
      
      # Update priority for the sampled data.
      replay_buffer.update_priorities(batch_index, td_error)
  
  sess.close()
  tf.contrib.keras.backend.clear_session()
  
  total_time = int(time.time() - start_time)
  print("Training finished in ", total_time, " s.", sep = "")
  
  # Close the environment.
  env.close()
  
  # Store data in a csv file.
  record = pd.DataFrame({"Step": list_step, "Mean Episodic Reward": list_mean_episodic_reward})
  record.to_csv(CSV_DIR + file_name + ".csv", sep = ",", index = False)

if __name__ == "__main__":
  """
  Default hyperparameter settings in train().
  ============================================================================================== 
  buffer_size = int(1e5)              # Size of the replay buffer
  alpha = 0.6                         # Temperature of the priority
  batch_size = 32                     # Minibatch size
  reward_min = -1.0                   # Lower bound of clipped reward
  reward_max = 1.0                    # Upper bound of clipped reward
  reward_discount = 0.99              # Discount factor
  
  epsilon_start = 1.0                 # Initial value for epsilon greedy policy
  epsilon_end = 0.05                  # Final value for epsilon greedy policy
  epsilon_decay_step = int(1e6)       # Time steps over which epsilon reaches its final value
  beta_start = 0.4                    # Initial temperature for importance sampling weights
  beta_end = 1.0                      # Final temperature for importance sampling weights
  beta_decay_step = int(1e7)          # Time steps over which beta reaches its final value
  lrs = [5e-5, 5e-6]                  # Learning rates at different stages
  lr_cutoff_steps = [int(8e6)]        # Time steps over which learning rate is annealed
  
  total_steps = int(1e7)              # Total frames
  initial_buffer_size = int(1e5)      # Initial buffer size before training begins
  target_network_update_step = 1000   # Interval between each target network update
  training_step = 4                   # Interval between each training step
  last_k_episodes = 100               # The episodic reward is averaged over the last k episodes
  print_frequency = 10                # Print training process every k episodes
  save_frequency = 0.01               # Chance of a better model is saved
  ============================================================================================== 
  """
  train(env_name = "SeaquestNoFrameskip-v4", file_name = "Seaquest_uniform", network_type = "conv", env_seed = 0, seed = 0, alpha = 0.0)
  train(env_name = "SeaquestNoFrameskip-v4", file_name = "Seaquest_prioritized", network_type = "conv", env_seed = 0, seed = 0)
  plot(input_file_names = ["Seaquest_uniform", "Seaquest_prioritized"], output_file_name = "Seaquest")