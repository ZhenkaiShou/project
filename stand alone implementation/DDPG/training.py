import csv
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from gym.envs.classic_control.pendulum import PendulumEnv

from config import *
from model import Actor, Critic
from noise import OUActionNoise
from replay_buffer import ReplayBuffer

def target_update_op(vars, target_vars, tau):
  # Update the target variables.
  update_op = [tf.assign(target_var, tau * var + (1 - tau) * target_var) for var, target_var in zip(vars, target_vars)]
  return update_op

def training(file_name):
  # Create folders.
  if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR)
  if not os.path.isdir(CSV_DIR):
    os.makedirs(CSV_DIR)
  if not os.path.isdir(FIGURE_TRAINING_DIR):
    os.makedirs(FIGURE_TRAINING_DIR)
  
  # Load models.
  actor = Actor(name = "actor")
  actor_target = Actor(name = "actor_target")
  actor_initial_update_op = target_update_op(actor.trainable_variables, actor_target.trainable_variables, 1.0)
  actor_target_update_op = target_update_op(actor.trainable_variables, actor_target.trainable_variables, TARGET_UPDATE_RATE)
  
  critic = Critic(name = "critic")
  critic.build_training()
  critic_target = Critic(name = "critic_target")
  critic_initial_update_op = target_update_op(critic.trainable_variables, critic_target.trainable_variables, 1.0)
  critic_target_update_op = target_update_op(critic.trainable_variables, critic_target.trainable_variables, TARGET_UPDATE_RATE)
  
  critic_with_actor = Critic(name = "critic", A = actor.pi)
  actor.build_training(critic_with_actor.actor_loss)
  
  env = PendulumEnv()
  replay_buffer = ReplayBuffer(BUFFER_SIZE)
  action_noise = OUActionNoise(np.zeros(A_LENGTH))
  
  with tf.Session() as sess:
    # Initialize actor and critic networks.
    sess.run(tf.global_variables_initializer())
    sess.run([actor_initial_update_op, critic_initial_update_op])
    
    list_final_reward = []
    
    additional_episode = int(np.ceil(MIN_BUFFER_SIZE / MAX_FRAME))
    for episode in range(-additional_episode, MAX_EPISODE):
      list_actor_loss = []
      list_critic_loss = []
      
      # Reset the environment and noise.
      s = env.reset()
      action_noise.reset()
      
      for step in range(MAX_FRAME):
        env.render()
        
        # Get action.
        a = sess.run(actor.pi, feed_dict = {actor.S: np.reshape(s, (1, -1))})
        noise = action_noise.get_noise()
        a = a[0] + ACTION_SCALING * noise
        a = np.clip(a, -ACTION_SCALING, ACTION_SCALING)
        
        # Interact with the game engine.
        s1, r, _, _ = env.step(a)
        
        # Add data to the replay buffer.
        data = [s, a, [r], s1]
        replay_buffer.append(data)
        
        if episode >= 0:
          for _ in range(BATCHES_PER_STEP):
            # Sample data from the replay buffer.
            batch_data = replay_buffer.sample(BATCH_SIZE)
            batch_s, batch_a, batch_r, batch_s1 = [np.array([batch_data[j][i] for j in range(BATCH_SIZE)]) for i in range(len(batch_data[0]))]
            
            # Compute the next action.
            a1 = sess.run(actor_target.pi, feed_dict = {actor_target.S: batch_s1})
            
            # Compute the target Q.
            q1 = sess.run(critic_target.q, feed_dict = {critic_target.S: batch_s1, critic_target.A: a1})
            q_target = batch_r + DISCOUNT * q1
            
            # Update actor and critic.
            _, _, actor_loss, critic_loss = sess.run([actor.train_op, critic.train_op, actor.actor_loss, critic.critic_loss], feed_dict = {actor.S: batch_s, critic_with_actor.S: batch_s, actor.LR: LR_ACTOR, critic.S: batch_s, critic.A: batch_a, critic.QTarget: q_target, critic.LR: LR_CRITIC})
            list_actor_loss.append(actor_loss)
            list_critic_loss.append(critic_loss)
            
            # Update target networks.
            sess.run([actor_target_update_op, critic_target_update_op])
        
        s = s1
      
      # Postprocessing after each episode.
      if episode >= 0:
        list_final_reward.append(r)
        avg_actor_loss = np.mean(list_actor_loss)
        avg_critic_loss = np.mean(list_critic_loss)
        print("Episode ", format(episode, "03d"), ":", sep = "")
        print("  Final Reward = ", format(r, ".6f"), ", Actor Loss = ", format(avg_actor_loss, ".6f"), ", Critic Loss = ", format(avg_critic_loss, ".6f"), sep = "") 
    
    # Testing.
    avg_reward = 0
    for i in range(TEST_EPISODE):
      # Reset the environment and noise.
      s = env.reset()
      action_noise.reset()
      
      for step in range(MAX_FRAME):
        env.render()
        
        # Get action.
        a = sess.run(actor.pi, feed_dict = {actor.S: np.reshape(s, (1, -1))})
        a = a[0]
        
        # Interact with the game engine.
        s, r, _, _ = env.step(a)
      
      # Postprocessing after each episode.
      avg_reward += r
    avg_reward /= TEST_EPISODE
    
    # Save the parameters.
    saver = tf.train.Saver([*actor.trainable_variables, *critic.trainable_variables])
    saver.save(sess, SAVE_DIR + file_name)
  tf.contrib.keras.backend.clear_session()
  env.close()
  
  # Store data in the csv file.
  with open(CSV_DIR + file_name + ".csv", "w") as f:
    fieldnames = ["Episode", "Final Reward", "Average Reward"]
    writer = csv.DictWriter(f, fieldnames = fieldnames, lineterminator = "\n")
    writer.writeheader()
    for episode in range(MAX_EPISODE):
      content = {"Episode": episode, "Final Reward": list_final_reward[episode]}
      if episode == MAX_EPISODE - 1:
        content.update({"Average Reward": avg_reward})
      writer.writerow(content)
  
  # Plot the training process.
  list_episode = list(range(MAX_EPISODE))
  f, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (5, 5))
  ax.plot(list_episode, list_final_reward, "r-", label = "Final Reward")
  ax.plot([MAX_EPISODE - 1], [avg_reward], "b.", label = "Average Reward")
  ax.set_title("Final Reward")
  ax.set_xlabel("Episode")
  ax.set_ylabel("Reward")
  ax.legend(loc = "lower right")
  ax.grid()
  
  f.savefig(FIGURE_TRAINING_DIR + file_name + ".png")
  plt.close(f)

def visualization(file_name):
  # Create folders.
  if not os.path.isdir(FIGURE_VISUALIZATION_DIR):
    os.makedirs(FIGURE_VISUALIZATION_DIR)
  
  # Load models.
  actor = Actor(name = "actor")
  critic = Critic(name = "critic")
  
  env = PendulumEnv()
  
  with tf.Session() as sess:
    # Load variables.
    saver = tf.train.Saver()
    saver.restore(sess, SAVE_DIR + file_name)
    
    list_obs = []
    list_reward = []
    s = env.reset()
    for step in range(MAX_FRAME):
      obs = env.render(mode = "rgb_array")
      list_obs.append(obs)
      
      # Get action.
      a = sess.run(actor.pi, feed_dict = {actor.S: np.reshape(s, (1, -1))})
      a = a[0]
      
      # Interact with the game engine.
      s, r, _, _ = env.step(a)
      list_reward.append(r)
  tf.contrib.keras.backend.clear_session()
  env.close()
  
  imageio.mimsave(FIGURE_VISUALIZATION_DIR + file_name + ".gif", [plot_obs(list_obs[i], list_reward[i]) for i in range(len(list_obs))], fps = 20)

def plot_obs(obs, reward):
  # Plot the observation.
  f, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (2, 2))
  ax.imshow(obs)
  ax.set_title("Reward: " + format(reward, ".8f"))
  ax.set_axis_off()
  
  f.canvas.draw()
  image = np.frombuffer(f.canvas.tostring_rgb(), dtype = "uint8")
  image = image.reshape(f.canvas.get_width_height()[::-1] + (3,))
  plt.close(f)

  return image

if __name__ == "__main__":
  # Function training(file_name):
  #
  # file_name determines the name of all saved files.
  training(file_name = "pendulum")
  
  # Function visualization(file_name):
  #
  # file_name determines which file from "Saves" folder will be used to restore the network variables.
  visualization(file_name = "pendulum")