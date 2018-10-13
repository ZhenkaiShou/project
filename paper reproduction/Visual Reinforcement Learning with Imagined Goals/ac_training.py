import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

import config
from ac import Actor, Critic
from reacher import MyReacherEnv
from replaybuffer import ReplayBuffer
from vae import VAE

# Local hyperparameters
Z_LENGTH = config.Z_LENGTH
A_LENGTH = config.A_LENGTH
ACTION_SCALING = config.ACTION_SCALING
EXPLORATION_ACTION_NOISE = 0.5
MAX_EXPLORATION_ACTION_NOISE = 1.0
TARGET_ACTION_NOISE = 0.2
MAX_TARGET_ACTION_NOISE = 0.5
MIXTURE = 0.5
C_REWARD = 1e-1
DISCOUNT = 0.99
BUFFER_SIZE = 8000
MIN_BUFFER_SIZE = 400
BATCHES_PER_STEP = 4
BATCH_SIZE = 100
TARGET_UPDATE_STEP = 2
TARGET_UPDATE_RATE = 0.01
VAE_UPDATE_STEP = 5
BATCHES_PER_TUNING = 5
MAX_EPISODE = 100000
MAX_FRAME = config.MAX_FRAME
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
LR_VAE = 1e-4

RAW_DATA_DIR = config.RAW_DATA_DIR
SAVE_VAE_DIR = config.SAVE_VAE_DIR
SAVE_AC_DIR = config.SAVE_AC_DIR
FIGURE_TRAINING_DIR = config.FIGURE_TRAINING_DIR

def target_update_op(vars, target_vars, tau):
  # Update the target variables.
  update_op = [tf.assign(target_var, tau * var + (1 - tau) * target_var) for var, target_var in zip(vars, target_vars)]
  return update_op

def ac_training(vae_file_name = "vae", ac_file_name = "ac"):
  # Create folders.
  if not os.path.isdir(SAVE_AC_DIR):
    os.makedirs(SAVE_AC_DIR)
  if not os.path.isdir(FIGURE_TRAINING_DIR):
    os.makedirs(FIGURE_TRAINING_DIR) 
  
  # Load training data for vae tuning.
  raw_data_list = os.listdir(RAW_DATA_DIR)
  raw_data = []
  for i in range(len(raw_data_list)):
    obs = np.load(RAW_DATA_DIR + raw_data_list[i])["obs"]
    raw_data.append(np.reshape(obs, (-1, 64, 64, 3)))
  raw_data = np.concatenate(raw_data, 0)
  
  # Load models.
  vae = VAE(name = "vae")
  vae.build_training()
    
  actor = Actor(name = "actor")
  actor_target = Actor(name = "actor_target")
  actor_initial_update_op = target_update_op(actor.trainable_variables, actor_target.trainable_variables, 1.0)
  actor_target_update_op = target_update_op(actor.trainable_variables, actor_target.trainable_variables, TARGET_UPDATE_RATE)
  
  critic1 = Critic(name = "critic")
  critic1.build_training()
  critic1_target = Critic(name = "critic_target")
  critic1_initial_update_op = target_update_op(critic1.trainable_variables, critic1_target.trainable_variables, 1.0)
  critic1_target_update_op = target_update_op(critic1.trainable_variables, critic1_target.trainable_variables, TARGET_UPDATE_RATE)
  
  critic2 = Critic(name = "critic2")
  critic2.build_training()
  critic2_target = Critic(name = "critic2_target")
  critic2_initial_update_op = target_update_op(critic2.trainable_variables, critic2_target.trainable_variables, 1.0)
  critic2_target_update_op = target_update_op(critic2.trainable_variables, critic2_target.trainable_variables, TARGET_UPDATE_RATE)
  
  critic1_with_actor = Critic(name = "critic", A = actor.pi)
  actor.build_training(critic1_with_actor.actor_loss)
  
  initialize_op = tf.global_variables_initializer()
  
  # Initialization.
  env = MyReacherEnv()
  replay_buffer = ReplayBuffer(BUFFER_SIZE)
  
  with tf.Session() as sess:
    # Initialize the Actor-Critic network parameters.
    sess.run(initialize_op)
    sess.run([actor_initial_update_op, critic1_initial_update_op, critic2_initial_update_op])
    
    saver_vae = tf.train.Saver(vae.trainable_variables)
    saver_ac = tf.train.Saver([*actor.trainable_variables, *critic1.trainable_variables, * critic2.trainable_variables])

    # Load VAE network parameters.
    saver_vae.restore(sess, SAVE_VAE_DIR + vae_file_name)
    
    list_final_distance = []
    
    # Compute additional episode for initializing the replay buffer.
    additional_episode = int(np.ceil(MIN_BUFFER_SIZE / MAX_FRAME))
    for episode in range(-additional_episode, MAX_EPISODE):
      temp_buffer = []
      list_actor_loss = []
      list_critic_loss = []
      
      # Reset the environment.
      env.reset()
      
      # Sample goals from prior N(0, 1).
      z_goal = np.random.randn(1, Z_LENGTH)
        
      # Get the first observation.
      obs = env.render(mode = "rgb_array")
      # Encode the observation.
      z = sess.run(vae.mu, feed_dict = {vae.Input: np.reshape(obs / 255.0, (-1, 64, 64, 3))})
        
      for step in range(MAX_FRAME):
        # Get the action.      
        a = sess.run(actor.pi, feed_dict = {actor.Z: z, actor.ZGoal: z_goal})
        noise = np.clip(EXPLORATION_ACTION_NOISE * np.random.randn(A_LENGTH), -MAX_EXPLORATION_ACTION_NOISE, MAX_EXPLORATION_ACTION_NOISE)
        a = np.clip(a[0] + ACTION_SCALING * noise, -ACTION_SCALING, ACTION_SCALING)
        
        # Interact with the game engine.
        env.step(a)
        
        # Get the next observation.
        obs_next = env.render(mode = "rgb_array")
        # Encode the next observation.
        z_next = sess.run(vae.mu, feed_dict = {vae.Input: np.reshape(obs_next / 255.0, (-1, 64, 64, 3))})
        
        # Add list [s, a, s', z', z_g] to the temporary buffer.
        temp_buffer.append([obs, a, obs_next, z_next[0], z_goal[0]])
        
        if episode >= 0:
          for _ in range(BATCHES_PER_STEP):
            # Sample data from the replay buffer.
            s_data = replay_buffer.sample(BATCH_SIZE)
            s_obs, s_a, s_obs_next, s_z_goal = [np.array([s_data[j][i] for j in range(BATCH_SIZE)]) for i in range(len(s_data[0]))]
            
            # Replace sampled goal with prior N(0, 1).
            random_number = np.random.uniform(size = (BATCH_SIZE, 1))
            random_z_goal = np.random.randn(BATCH_SIZE, Z_LENGTH)
            s_z_goal = np.where(random_number < MIXTURE, s_z_goal, random_z_goal)
            
            # Encode the observations and compute the reward.
            s_z = sess.run(vae.mu, feed_dict = {vae.Input: s_obs / 255.0})
            s_z_next = sess.run(vae.mu, feed_dict = {vae.Input: s_obs_next / 255.0})
            r = -C_REWARD * np.linalg.norm(s_z_next - s_z_goal, axis = 1, keepdims = True)
            
            # We now have all the sampled data (z, a, z', r, z_g) for training.
            
            # Compute the next action.
            s_a_next = sess.run(actor_target.pi, feed_dict = {actor_target.Z: s_z_next, actor_target.ZGoal: s_z_goal})
            noise = np.clip(TARGET_ACTION_NOISE * np.random.randn(BATCH_SIZE, A_LENGTH), -MAX_TARGET_ACTION_NOISE, MAX_TARGET_ACTION_NOISE)
            s_a_next = np.clip(s_a_next + ACTION_SCALING * noise, -ACTION_SCALING, ACTION_SCALING)
            
            # Compute the target Q.
            q1 = sess.run(critic1_target.q, feed_dict = {critic1_target.Z: s_z_next, critic1_target.A: s_a_next, critic1_target.ZGoal: s_z_goal})
            q2 = sess.run(critic2_target.q, feed_dict = {critic2_target.Z: s_z_next, critic2_target.A: s_a_next, critic2_target.ZGoal: s_z_goal})
            q_target = r + DISCOUNT * np.minimum(q1, q2)
            
            # Update critics.
            _, critic_loss = sess.run([critic1.train_op, critic1.critic_loss], feed_dict = {critic1.Z: s_z, critic1.A: s_a, critic1.ZGoal: s_z_goal, critic1.QTarget: q_target, critic1.LR: LR_CRITIC})
            sess.run(critic2.train_op, feed_dict = {critic2.Z: s_z, critic2.A: s_a, critic2.ZGoal: s_z_goal, critic2.QTarget: q_target, critic2.LR: LR_CRITIC})
            list_critic_loss.append(critic_loss)
            
          if step % TARGET_UPDATE_STEP == 0:
            # Update actor.
            _, actor_loss = sess.run([actor.train_op, actor.actor_loss], feed_dict = {actor.Z: s_z, actor.ZGoal: s_z_goal, critic1_with_actor.Z: s_z, critic1_with_actor.ZGoal: s_z_goal, actor.LR: LR_ACTOR})
            list_actor_loss.append(actor_loss)
            
            # Update target networks.
            sess.run([actor_target_update_op, critic1_target_update_op, critic2_target_update_op])
        
        obs = obs_next
        z = z_next
      
      # Postprocessing after an episode ends.
      for i in range(MAX_FRAME):
        # Relabel the goal with future strategy.
        # Temp buffer = [s, a, s', z', z_g].
        obs, a, obs_next, _, _ = temp_buffer[i]
        # Sample future index between i and (i+8).
        future_index = np.random.randint(i, np.minimum(i+8, MAX_FRAME))
        _, _, _, new_goal, _ = temp_buffer[future_index]
        # Add list [s, a, s', z_g] to the replay buffer.
        replay_buffer.append([obs, a, obs_next, new_goal])
        
      if episode >= 0:
        if episode % VAE_UPDATE_STEP == 0 and episode > 0:
          # Update vae.
          for batch in range(BATCHES_PER_TUNING):
            # Sample from raw data and replay buffer.
            random_index = np.random.choice(len(raw_data), BATCH_SIZE // 2, replace = False)
            sample_raw_data = raw_data[random_index]
            sample_replay_buffer = replay_buffer.sample(BATCH_SIZE // 2)
            sample_replay_buffer = np.array([sample_replay_buffer[i][0] for i in range(BATCH_SIZE // 2)])
            sample_obs = np.concatenate([sample_raw_data, sample_replay_buffer], 0)
            np.random.shuffle(sample_obs)
            
            # Fine tune vae.
            sess.run(vae.train_op, feed_dict = {vae.Input: sample_obs / 255.0, vae.LR: LR_VAE})
          
          # Save the parameters.
          saver_vae.save(sess, SAVE_VAE_DIR + vae_file_name + "_tuning")
          saver_ac.save(sess, SAVE_AC_DIR + ac_file_name)
        
        final_distance = C_REWARD * np.linalg.norm(z_next - z_goal, axis = 1)[0]
        list_final_distance.append(final_distance)
        avg_actor_loss = np.mean(list_actor_loss)
        avg_critic_loss = np.mean(list_critic_loss)
        print("Episode ", format(episode, "05d"), ":", sep = "")
        print("  Final Distance = ", format(final_distance, ".8f"),  ", Actor Loss = ", format(avg_actor_loss, ".8f"), ", Critic Loss = ", format(avg_critic_loss, ".8f"), sep = "")
    
    # Save the parameters.
    saver_ac.save(sess, SAVE_AC_DIR + ac_file_name)
  
  tf.contrib.keras.backend.clear_session()
  env.close()
  
  # Plot the training process.
  list_episode = list(range(MAX_EPISODE))
  f, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (5, 5))
  ax.plot(list_episode, list_final_distance, "r-")
  ax.set_title("Final Distance")
  ax.set_xlabel("Episode")
  ax.set_ylabel("Distance")
  ax.grid()
  
  f.savefig(FIGURE_TRAINING_DIR + ac_file_name + ".png")
  plt.close(f)

if __name__ == "__main__":
  ac_training()