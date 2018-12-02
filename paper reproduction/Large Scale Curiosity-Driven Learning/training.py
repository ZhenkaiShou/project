import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

import config
from average_mean_std import average_mean_std
from dynamics import Dynamics
from parallel_environment import ParallelEnvironment
from policy import Policy
from wrappers import make_atari

# Local hyperparameters.
ENV_NAME = config.ENV_NAME
AUXILIARY_TASK = config.AUXILIARY_TASK
AUTOSAVE_STEP = 10
RANDOM_STEP = 10000
NUM_ENV = 32
EPOCH = 3
BATCH_SIZE = 16
TIME_STEP_PER_UPDATE = 128
MAX_FRAME = 1e8
COEF_EXT_REWARD = 0.0
COEF_INT_REWARD = 1.0
GAMMA = 0.99
LAMBDA = 0.95

SAVE_DIR = config.SAVE_DIR
FIGURE_DIR = config.FIGURE_DIR

def train():
  save_path = SAVE_DIR + ENV_NAME + "/" + AUXILIARY_TASK + "/"
  figure_path = FIGURE_DIR + ENV_NAME + "/" + AUXILIARY_TASK + "/"
  # Create folders.
  if not os.path.isdir(save_path):
    os.makedirs(save_path)
  if not os.path.isdir(figure_path):
    os.makedirs(figure_path)
  
  # Get observation space and action space.
  env = make_atari(ENV_NAME)
  obs_space = env.observation_space
  action_space = env.action_space
  
  # Estimate the mean and standard deviation of observations.
  env.reset()
  list_obs = []
  for _ in range(RANDOM_STEP):
    action = action_space.sample()
    obs, _, done, _ = env.step(action)
    if done:
      obs = env.reset()
    list_obs.append(obs)
  obs_mean = np.mean(list_obs, 0)
  obs_std = np.mean(np.std(list_obs, 0))
  np.savez_compressed(save_path + "obs_mean_std", obs_mean = obs_mean, obs_std = obs_std)
  env.close()
  del env
  
  # Build models.
  dynamics = Dynamics(obs_space, action_space, auxiliary_task = AUXILIARY_TASK, is_training = True)
  policy = Policy(obs_space, action_space, is_training = True)
  
  variables_initializer = tf.global_variables_initializer()
  
  # Create environments.
  par_env = ParallelEnvironment([make_atari(ENV_NAME) for _ in range(NUM_ENV)])
  
  with tf.Session() as sess:
    # Initialize variables.
    sess.run(variables_initializer)
    
    saver_dynamics = tf.train.Saver(dynamics.trainable_variables)
    saver_policy = tf.train.Saver(policy.trainable_variables)
    
    # Initialize the running estimate of rewards.
    sum_reward = np.zeros(NUM_ENV)
    reward_mean = 0.0
    reward_std = 1.0
    reward_count = 0
    
    total_rollout_step = 0
    total_update_step = 0
    total_frame = 0
    sum_ext_reward = np.zeros((NUM_ENV, TIME_STEP_PER_UPDATE))
    list_highest_reward = []
    num_batch = int(np.ceil(NUM_ENV / BATCH_SIZE))
    
    # Each while loop is a rollout step, which first interacts with the environment and then updates the network.
    while total_frame <= MAX_FRAME:
      # Initialize buffers.
      buffer_obs = np.zeros((NUM_ENV, TIME_STEP_PER_UPDATE + 1, *obs_space.shape))
      buffer_action = np.zeros((NUM_ENV, TIME_STEP_PER_UPDATE))
      buffer_ext_reward = np.zeros((NUM_ENV, TIME_STEP_PER_UPDATE))
      buffer_done = np.zeros((NUM_ENV, TIME_STEP_PER_UPDATE))
      buffer_log_prob = np.zeros((NUM_ENV, TIME_STEP_PER_UPDATE))
      buffer_v = np.zeros((NUM_ENV, TIME_STEP_PER_UPDATE + 1))
      buffer_int_reward = np.zeros((NUM_ENV, TIME_STEP_PER_UPDATE))
      buffer_reward = np.zeros((NUM_ENV, TIME_STEP_PER_UPDATE))
      buffer_sum_reward = np.zeros((NUM_ENV, TIME_STEP_PER_UPDATE))
      buffer_adv = np.zeros((NUM_ENV, TIME_STEP_PER_UPDATE))
      buffer_v_target = np.zeros((NUM_ENV, TIME_STEP_PER_UPDATE))
      
      # Interact with the environment for TIME_STEP_PER_UPDATE steps.
      for step in range(TIME_STEP_PER_UPDATE):
        # Get observation.
        if total_frame == 0:
          obs = par_env.reset()
        else:
          obs, _, _, _ = par_env.get_last_response()
        obs = (obs - obs_mean) / obs_std
        # Sample action.
        action, log_prob, v = sess.run([policy.sampled_action, policy.sampled_log_prob, policy.v], feed_dict = {policy.Obs: np.expand_dims(obs, 1)})
        action = np.squeeze(action, 1)
        log_prob = np.squeeze(log_prob, 1)
        v = np.squeeze(v, 1)
        
        # Interact with the environment.
        obs_next, extrinsic_reward, done, _ = par_env.step(action)
        
        # Update buffers.
        buffer_obs[:, step] = obs
        buffer_action[:, step] = action
        buffer_ext_reward[:, step] = extrinsic_reward
        buffer_done[:, step] = done
        buffer_log_prob[:, step] = log_prob
        buffer_v[:, step] = v
        
        if step == TIME_STEP_PER_UPDATE - 1:
          # Extra operations for the last time step.
          obs_next = (obs_next - obs_mean) / obs_std
          v_next = sess.run(policy.v, feed_dict = {policy.Obs: np.expand_dims(obs_next, 1)})
          v_next = np.squeeze(v_next, 1)
          buffer_obs[:, step + 1] = obs_next
          buffer_v[:, step + 1] = v_next
        
        # Update frame counter.
        total_frame += NUM_ENV
      
      # Get the highest reward.
      for step in range(TIME_STEP_PER_UPDATE):
        sum_ext_reward[:, step] = buffer_ext_reward[:, step] + (1 - buffer_done[:, step-1]) * sum_ext_reward[:, step-1]
      highest_reward = np.amax(sum_ext_reward)
      list_highest_reward.append(highest_reward)
      
      # Compute the intrinsic reward.
      buffer_int_reward[:] = sess.run(dynamics.intrinsic_reward, 
        feed_dict = {dynamics.Obs: buffer_obs[:, :-1], dynamics.ObsNext: buffer_obs[:, 1:], dynamics.Action: buffer_action})
      # The total reward is a mixture of extrinsic reward and intrinsic reward.
      buffer_reward[:] = COEF_EXT_REWARD * np.clip(buffer_ext_reward, -1.0, 1.0) + COEF_INT_REWARD * buffer_int_reward
      
      # Normalize reward by dividing it by a running estimate of the standard deviation of the sum of discounted rewards.
      # 1. Compute the sum of discounted rewards.
      for step in range(TIME_STEP_PER_UPDATE):
        sum_reward = buffer_reward[:, step] + GAMMA * sum_reward
        buffer_sum_reward[:, step] = sum_reward
      # 2. Compute mean and standard deviation of the sum of discounted rewards.
      reward_batch_mean = np.mean(buffer_sum_reward)
      reward_batch_std = np.std(buffer_sum_reward)
      reward_batch_count = np.size(buffer_sum_reward)
      # 3. Update the running estimate of standard deviation.
      reward_mean, reward_std, reward_count = average_mean_std(reward_mean, reward_std, reward_count, reward_batch_mean, reward_batch_std, reward_batch_count)
      # 4. Normalize reward.
      buffer_reward = buffer_reward / reward_std
      
      # Compute advantage.
      # - gae_adv_t = sum((gamma * lambda)^i * adv_(t+l)) over i in [0, inf)
      # - adv_t = r_t + gamma * v_(t+1) - v_t
      adv = buffer_reward + GAMMA * buffer_v[:, 1:] - buffer_v[:, :-1]
      sum_adv = np.zeros(NUM_ENV)
      for step in range(TIME_STEP_PER_UPDATE - 1, -1, -1):
        sum_adv = adv[:, step] + GAMMA * LAMBDA * sum_adv
        buffer_adv[:, step] = sum_adv
      
      # Compute target value.
      buffer_v_target[:] = buffer_adv + buffer_v[:, :-1]
      
      # Normalize advantage with zero mean and unit variance.
      adv_mean = np.mean(buffer_adv)
      adv_std = np.std(buffer_adv)
      buffer_adv = (buffer_adv - adv_mean) / adv_std
      
      # Update networks.
      for epoch in range(EPOCH):
        random_id = np.arange(NUM_ENV)
        np.random.shuffle(random_id)
        for i in range(num_batch):
          batch_id = random_id[i * BATCH_SIZE: np.minimum(NUM_ENV, (i+1) * BATCH_SIZE)] 
          _, auxiliary_loss, dyna_loss = sess.run([dynamics.train_op, dynamics.auxiliary_loss, dynamics.dyna_loss],
            feed_dict = {dynamics.Obs: buffer_obs[:, :-1], dynamics.ObsNext: buffer_obs[:, 1:], dynamics.Action: buffer_action})
          _, value_loss, pg_loss, entropy_loss = sess.run([policy.train_op, policy.value_loss, policy.pg_loss, policy.entropy_loss], 
            feed_dict = {policy.Obs: buffer_obs[:, :-1], policy.Action: buffer_action, 
            policy.Adv: buffer_adv, policy.VTarget: buffer_v_target, policy.LogProbOld: buffer_log_prob})
          total_update_step += 1
      
      # Update rollout step.
      total_rollout_step += 1
      
      # Only print the last update step.
      print("Rollout Step ", total_rollout_step, ", Total Frame ", total_frame, ", Update Step ", total_update_step, ":", sep = "")
      print("  Auxiliary Loss = ", format(auxiliary_loss, ".6f"), ", Dynamics Loss = ", format(dyna_loss, ".6f"), 
        ", Value Loss = ", format(value_loss, ".6f"), ", Policy Loss = ", format(pg_loss, ".6f"), sep = "")
      print("  Highest Reward = ", highest_reward, sep = "")
      
      if total_rollout_step % AUTOSAVE_STEP == 0:
        # Save network parameters.
        saver_dynamics.save(sess, save_path + "dynamics")
        saver_policy.save(sess, save_path + "policy")
        # Plot reward.
        interval = NUM_ENV * TIME_STEP_PER_UPDATE
        list_frame = list(range(interval, (total_rollout_step+1) * interval, interval))
        plot_reward(list_frame, list_highest_reward, figure_path)
    
    # Save network parameters.
    saver_dynamics.save(sess, save_path + "dynamics")
    saver_policy.save(sess, save_path + "policy")
    # Plot reward.
    interval = NUM_ENV * TIME_STEP_PER_UPDATE
    list_frame = list(range(interval, total_frame + interval, interval))
    plot_reward(list_frame, list_highest_reward, figure_path)
  par_env.close()
  
def plot_reward(list_frame, list_highest_reward, figure_path):
  # Plot the sum of extrinsic reward over the whole training process.
  f, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (5, 5))
  ax.plot(list_frame, list_highest_reward, "r-")
  ax.set_xlabel("Frames")
  ax.set_ylabel("Extrinsic Reward per Episode")
  ax.ticklabel_format(style = "sci", axis = "x", scilimits = (0, 0))
  ax.grid()
  f.savefig(figure_path + "extrinsic_reward.png")
  plt.close(f)
  
if __name__ == "__main__":
  train()
