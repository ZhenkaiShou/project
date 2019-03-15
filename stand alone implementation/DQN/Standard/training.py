import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import time

from atari_wrappers import make_atari
from config import *
from model import QValueNetwork
from replay_buffer import ReplayBuffer

def plot_episodic_reward(list_episodic_reward, file_name):
  list_frame, list_reward = [[list_episodic_reward[j][i] for j in range(len(list_episodic_reward))] for i in range(len(list_episodic_reward[0]))]
  f, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (5, 5))
  ax.plot(list_frame, list_reward)
  ax.set_title("Episodic reward")
  ax.set_xlabel("Frame")
  ax.set_ylabel("Episodic Reward")
  ax.ticklabel_format(style = "sci", axis = "x", scilimits = (0, 0))
  ax.grid()
  
  f.savefig(FIGURE_TRAINING_DIR + file_name + ".png")
  plt.close(f)

def train(file_name):
  # Create folders.
  if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR)
  if not os.path.isdir(FIGURE_TRAINING_DIR):
    os.makedirs(FIGURE_TRAINING_DIR)
  
  # Obtain environment parameters.
  env = make_atari(ENV_NAME)
  obs_space = env.observation_space
  action_space = env.action_space
  
  # Build networks.
  main_network = QValueNetwork(obs_space, action_space, name = "main_network")
  target_network = QValueNetwork(obs_space, action_space, name = "target_network", auxiliary_network = main_network)
  variables_initializer = tf.global_variables_initializer()
  
  replay_buffer = ReplayBuffer(buffer_size = BUFFER_SIZE)
  start_time = time.time()
  list_episodic_reward = []
  episodic_reward = 0
  
  obs = env.reset()
  
  with tf.Session() as sess:
    # Initialize all variables.
    sess.run(variables_initializer)
    # Only save the main network.
    saver = tf.train.Saver(var_list = main_network.variables)
    
    # Initialize buffers.
    for _ in range(INITIAL_BUFFER_SIZE):
      # Sample random action.
      action = np.random.randint(action_space.n)
      # Interact with the environment.
      obs_next, reward, done, _ = env.step(action)
      episodic_reward += reward
      if done:
        obs_next = env.reset()
        episodic_reward = 0
      # Store data.
      data = [obs, action, reward, done, obs_next]
      replay_buffer.append(data)
      # Update observation.
      obs = obs_next
    
    for step in range(TOTAL_STEP):
      # Synchronize the target network periodically (target network <- main network).
      if step % TARGET_NETWORK_UPDATE_STEP == 0:
        sess.run(target_network.sync_op)
      
      # Sample action with epsilon-greedy policy.
      epsilon = EPSILON_MAX - (EPSILON_MAX - EPSILON_MIN) * np.minimum(step / EPSILON_DECAY_STEP, 1)
      if np.random.uniform() < epsilon:
        action = np.random.randint(action_space.n)
      else:
        q = sess.run(target_network.q, feed_dict = {target_network.Obs: np.expand_dims(np.array(obs) / 255.0, 0)})
        action = np.argmax(q[0])
      # Interact with the environment.
      obs_next, reward, done, _ = env.step(action)
      episodic_reward += reward
      if done:
        obs_next = env.reset()
        list_episodic_reward.append((step, episodic_reward))
        delta_time = int(time.time() - start_time)
        print("Step ", step, "/", TOTAL_STEP, ": Time spent = ", delta_time, " s , Episodic reward = ", episodic_reward, sep = "")
        episodic_reward = 0
      # Store data.
      data = [obs, action, reward, done, obs_next]
      replay_buffer.append(data)
      # Update observation.
      obs = obs_next
      
      # Learning rate.
      lr = LEARNING_RATE[-1]
      for i in range(len(LR_ANNEAL_STEP)):
        if step < LR_ANNEAL_STEP[i]:
          lr = LEARNING_RATE[i]
          break
      
      # Sample training data from the replay buffer.
      batch_data = replay_buffer.sample(BATCH_SIZE)
      batch_obs, batch_action, batch_reward, batch_done, batch_obs_next = \
        [np.array([batch_data[j][i] for j in range(BATCH_SIZE)]) for i in range(len(batch_data[0]))]
      
      # Compute the target Q value:
      #   target_q = r + (1 - done) * REWARD_DISCOUNT * max[q(s', a)]
      q_next = sess.run(target_network.q, feed_dict = {target_network.Obs: batch_obs_next / 255.0})
      max_qnext = np.amax(q_next, axis = 1)
      target_q = batch_reward + (1 - batch_done) * REWARD_DISCOUNT * max_qnext
      
      # Update the main network.
      sess.run(main_network.train_op, feed_dict = {
        main_network.Obs: batch_obs / 255.0, main_network.Action: batch_action, main_network.TargetQ: target_q, main_network.LR: lr
        })
      
      # Save the main network periodically.
      if step % AUTOSAVE_STEP == 0:
        saver.save(sess, SAVE_DIR + file_name)
    
    # Save the main network.
    saver = tf.train.Saver(var_list = main_network.variables)
    saver.save(sess, SAVE_DIR + file_name)
  
  total_time = int(time.time() - start_time)
  print("Training finished in ", total_time, " s.", sep = "")
  
  # Close the environment.
  env.close()
  
  # Plot the episodic reward against training step curve.
  plot_episodic_reward(list_episodic_reward, file_name)

if __name__ == "__main__":
  train(file_name = "dqn")