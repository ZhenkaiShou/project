import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

import config
from ac import Actor, Critic
from reacher import MyReacherEnv
from replaybuffer import ReplayBuffer
from rnn import RNN
from vae import VAE

# Local hyperparameters
Z_LENGTH = config.Z_LENGTH
A_LENGTH = config.A_LENGTH
ACTION_SCALING = config.ACTION_SCALING
EPSILON_DECAY = 0.9995
MIN_EPSILON = 0.75
EXPLORATION_ACTION_NOISE = 0.5
MAX_EXPLORATION_ACTION_NOISE = 1.0
TARGET_ACTION_NOISE = 0.1
MAX_TARGET_ACTION_NOISE = 0.2
MIXTURE = 1.0
FUTURE_SAMPLES = 4
REWARD_SCALING = config.REWARD_SCALING
DISCOUNT = 0.99
BUFFER_SIZE = 100000
MIN_BUFFER_SIZE = 2000
BATCHES_PER_STEP = 4
BATCH_SIZE = 100
TARGET_UPDATE_STEP = 2
TARGET_UPDATE_RATE = 1.0
MAX_EPISODE = 2000
MAX_FRAME = config.MAX_FRAME
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4

SAVE_VAE_DIR = config.SAVE_VAE_DIR
SAVE_RNN_DIR = config.SAVE_RNN_DIR
SAVE_AC_DIR = config.SAVE_AC_DIR
FIGURE_TRAINING_DIR = config.FIGURE_TRAINING_DIR

def target_update_op(vars, target_vars, tau):
  # Update the target variables.
  update_op = [tf.assign(target_var, tau * var + (1 - tau) * target_var) for var, target_var in zip(vars, target_vars)]
  return update_op

def ac_training(vae_file_name = "vae", rnn_file_name = "rnn", ac_file_name = "ac"):
  # Create folders.
  if not os.path.isdir(SAVE_AC_DIR):
    os.makedirs(SAVE_AC_DIR)
  if not os.path.isdir(FIGURE_TRAINING_DIR):
    os.makedirs(FIGURE_TRAINING_DIR)
  
  # Load models.
  vae = VAE(name = "vae")
  rnn = RNN(name = "rnn", is_single_input = True)
  
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
  epsilon = 1.0
  
  with tf.Session() as sess:
    # Initialize the Actor-Critic network parameters.
    sess.run(initialize_op)
    sess.run([actor_initial_update_op, critic1_initial_update_op, critic2_initial_update_op])
    
    saver_vae = tf.train.Saver(vae.trainable_variables)
    saver_rnn = tf.train.Saver(rnn.trainable_variables)
    saver_ac = tf.train.Saver([*actor.trainable_variables, *critic1.trainable_variables, * critic2.trainable_variables])

    # Load VAE and RNN network parameters.
    saver_vae.restore(sess, SAVE_VAE_DIR + vae_file_name)
    saver_rnn.restore(sess, SAVE_RNN_DIR + rnn_file_name)
    
    list_final_distance = []
    
    # Compute additional episode for initializing the replay buffer.
    additional_episode = int(np.ceil(MIN_BUFFER_SIZE / MAX_FRAME / (1 + FUTURE_SAMPLES)))
    for episode in range(-additional_episode, MAX_EPISODE):
      temp_buffer = []
      list_actor_loss = []
      list_critic_loss = []
      
      # Get intitial state of RNN.
      state = sess.run(rnn.initial_state)
      
      # Sample goals from the game engine.
      env.reset()
      goal = env.render(mode = "rgb_array")
      z_goal = sess.run(vae.mu, feed_dict = {vae.Input: np.reshape(goal / 255.0, (1, *goal.shape))})
      
      # Reset the environment.
      env.reset()
      
      # Sample goals from prior N(0, 1), not working.
      #z_goal = np.random.randn(1, Z_LENGTH)
      
      # Get the first observation.
      obs = env.render(mode = "rgb_array")
      while np.linalg.norm(obs - goal) == 0:
        obs = env.render(mode = "rgb_array")
      # Encode the observation.
      z = sess.run(vae.mu, feed_dict = {vae.Input: np.reshape(obs / 255.0, (1, *obs.shape))})
        
      for step in range(MAX_FRAME):
        # Get the action with epsilon-greedy policy.
        random_number = np.random.uniform()
        if random_number <= epsilon:
          noise = np.clip(EXPLORATION_ACTION_NOISE * np.random.randn(1, A_LENGTH), -MAX_EXPLORATION_ACTION_NOISE, MAX_EXPLORATION_ACTION_NOISE)
          a = ACTION_SCALING * noise
        else:
          a = sess.run(actor.pi, feed_dict = {actor.Z: z, actor.H: np.concatenate([state.h, state.c], 1), actor.ZGoal: z_goal})
        
        # Update the hidden state.
        state_next = sess.run(rnn.final_state, feed_dict = {rnn.Z: np.reshape(z, (1, *z.shape)), rnn.A: np.reshape(a, (1, *a.shape)), rnn.initial_state: state})
        
        # Interact with the game engine.
        env.step(a[0])
        
        # Get the next observation.
        obs_next = env.render(mode = "rgb_array")
        # Encode the next observation.
        z_next = sess.run(vae.mu, feed_dict = {vae.Input: np.reshape(obs_next / 255.0, (1, *obs.shape))})
        
        # Add list [z, h, a, z', h', z_g] to the temporary buffer.
        temp_buffer.append([z[0], np.concatenate([state.h, state.c], 1)[0], a[0], z_next[0], np.concatenate([state_next.h, state_next.c], 1)[0], z_goal[0]])
        
        if episode >= 0:
          for _ in range(BATCHES_PER_STEP):
            # Sample data from the replay buffer.
            s_data = replay_buffer.sample(BATCH_SIZE)
            s_z, s_h, s_a, s_z_next, s_h_next, s_z_goal = [np.array([s_data[j][i] for j in range(BATCH_SIZE)]) for i in range(len(s_data[0]))]
            
            # Replace 0% of sampled goals with prior N(0, 1).
            # Change MIXTURE to 0.5 to replace half of the goals with imagination.
            random_number = np.random.uniform(size = (BATCH_SIZE, 1))
            random_z_goal = np.random.randn(BATCH_SIZE, Z_LENGTH)
            s_z_goal = np.where(random_number < MIXTURE, s_z_goal, random_z_goal)
            
            # Compute the reward.
            r = -REWARD_SCALING * np.linalg.norm(s_z_next - s_z_goal, axis = 1, keepdims = True)
            
            # We now have all the sampled data (z, h, a, z', h', r, z_g) for training.
            
            # Compute the next action.
            s_a_next = sess.run(actor_target.pi, feed_dict = {actor_target.Z: s_z_next, actor_target.H: s_h_next, actor_target.ZGoal: s_z_goal})
            noise = np.clip(TARGET_ACTION_NOISE * np.random.randn(BATCH_SIZE, A_LENGTH), -MAX_TARGET_ACTION_NOISE, MAX_TARGET_ACTION_NOISE)
            s_a_next = np.clip(s_a_next + ACTION_SCALING * noise, -ACTION_SCALING, ACTION_SCALING)
            
            # Compute the target Q.
            q1 = sess.run(critic1_target.q, feed_dict = {critic1_target.Z: s_z_next, critic1_target.H: s_h_next, critic1_target.A: s_a_next, critic1_target.ZGoal: s_z_goal})
            q2 = sess.run(critic2_target.q, feed_dict = {critic2_target.Z: s_z_next, critic2_target.H: s_h_next, critic2_target.A: s_a_next, critic2_target.ZGoal: s_z_goal})
            q_target = r + DISCOUNT * np.minimum(q1, q2)
            
            # Update critics.
            _, critic_loss = sess.run([critic1.train_op, critic1.critic_loss], feed_dict = {critic1.Z: s_z, critic1.H: s_h, critic1.A: s_a, critic1.ZGoal: s_z_goal, critic1.QTarget: q_target, critic1.LR: LR_CRITIC})
            sess.run(critic2.train_op, feed_dict = {critic2.Z: s_z, critic2.H: s_h, critic2.A: s_a, critic2.ZGoal: s_z_goal, critic2.QTarget: q_target, critic2.LR: LR_CRITIC})
            list_critic_loss.append(critic_loss)
            
          if step % TARGET_UPDATE_STEP == 0:
            # Update actor.
            _, actor_loss = sess.run([actor.train_op, actor.actor_loss], feed_dict = {actor.Z: s_z, actor.H: s_h, actor.ZGoal: s_z_goal, critic1_with_actor.Z: s_z, critic1_with_actor.H: s_h, critic1_with_actor.ZGoal: s_z_goal, actor.LR: LR_ACTOR})
            list_actor_loss.append(actor_loss)
        
        obs = obs_next
        z = z_next
        state = state_next
      
      # Relabel the goal with future strategy.
      for i in range(MAX_FRAME):
        # Temp buffer = [z, h, a, z', h', z_g].
        temp_z, temp_h, temp_a, temp_z_next, temp_h_next, temp_z_goal = temp_buffer[i]
        replay_buffer.append([temp_z, temp_h, temp_a, temp_z_next, temp_h_next, temp_z_goal])
        # Sample future index between [i, MAX_FRAME).
        for _ in range(FUTURE_SAMPLES):
          future_index = np.random.randint(i, MAX_FRAME)
          _, _, _, new_goal, _, _ = temp_buffer[future_index]
          # Add list [z, h, a, z', h', z_g] to the replay buffer.
          replay_buffer.append([temp_z, temp_h, temp_a, temp_z_next, temp_h_next, new_goal])
      
      # Postprocessing after an episode ends.
      if episode >= 0:
        # Update target networks.
        sess.run([actor_target_update_op, critic1_target_update_op, critic2_target_update_op])
        
        # Update epsilon.
        epsilon = np.maximum(epsilon * EPSILON_DECAY, MIN_EPSILON)
        
        final_distance = REWARD_SCALING * np.linalg.norm(z_next - z_goal, axis = 1)[0]
        list_final_distance.append(final_distance)
        avg_actor_loss = np.mean(list_actor_loss)
        avg_critic_loss = np.mean(list_critic_loss)
        print("Episode ", format(episode, "04d"), ":", sep = "")
        print("  Final Distance = ", format(final_distance, ".8f"),  ", Actor Loss = ", format(avg_actor_loss, ".8f"), ", Critic Loss = ", format(avg_critic_loss, ".8f"), sep = "")
        
        if episode % 100 == 0:
          # Save the parameters.
          saver_ac.save(sess, SAVE_AC_DIR + ac_file_name)
      
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