import gym
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import time

from atari_wrappers import make_atari
from config import *
from model import QValueNetwork
from replay_buffer import ReplayBuffer

start_time = time.time()

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

class Worker(object):
  def __init__(self,  main_network, thread_id):
    self.main_network = main_network
    self.thread_id = thread_id
    
    # Create environment.
    self.env = make_atari(ENV_NAME)
    self.obs_space = self.env.observation_space
    self.action_space = self.env.action_space
    
    # Local network.
    self.local_network = QValueNetwork(self.obs_space, self.action_space, name = "local_network_" + str(thread_id), auxiliary_network = main_network)
    # Target network.
    self.target_network = QValueNetwork(self.obs_space, self.action_space, name = "target_network_" + str(thread_id), auxiliary_network = main_network)
    
def worker_process(job_name, task_index, cluster_dict, file_name):
  import tensorflow as tf
  # GPU training.
  if USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = PER_PROCESS_GPU_MEMORY_FRACTION)
    config = tf.ConfigProto(gpu_options = gpu_options)
  else:
    config = None
  
  # Create and start a server for the local task.
  cluster = tf.train.ClusterSpec(cluster_dict)
  server = tf.train.Server(cluster, job_name = job_name, task_index = task_index, config = config)
  
  if job_name == "ps":
    # Parameter server.
    with tf.device("/job:" + job_name + "/task:" + str(task_index)):
      queue = tf.FIFOQueue(cluster.num_tasks("worker"), tf.int32, shared_name = "done_queue" + str(task_index))
    # Close the parameter server when all queues from workers have been filled.
    with tf.Session(server.target) as sess:
      for i in range(cluster.num_tasks("worker")):
        sess.run(queue.dequeue())
    return []
  
  elif job_name == "worker":
    # Obtain environment parameters.
    env = make_atari(ENV_NAME)
    obs_space = env.observation_space
    action_space = env.action_space
    
    # Worker.
    with tf.device(tf.train.replica_device_setter(worker_device = "/job:" + job_name + "/task:" + str(task_index), cluster = cluster)):
      # Build networks.
      main_network = QValueNetwork(obs_space, action_space, name = "main_network")
      target_network = QValueNetwork(obs_space, action_space, name = "target_network", auxiliary_network = main_network)
    
    replay_buffer = ReplayBuffer(buffer_size = BUFFER_SIZE)
    list_episodic_reward = []
    episodic_reward = 0
    obs = env.reset()
    
    # Additional settings for the first worker (task_index = 0).
    if task_index == 0:
      saver = tf.train.Saver(var_list = main_network.variables, max_to_keep = 1)
      next_target_network_update_step = 0
      next_autosave_step = 0
    
    with tf.train.MonitoredTrainingSession(
      master = server.target,
      is_chief = (task_index == 0),
      config = config,
      save_summaries_steps = None,
      save_summaries_secs = None,
      save_checkpoint_steps = None,
      save_checkpoint_secs = None
      ) as sess:
      
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
      
      # Run until reaching maximum training steps.
      while sess.run(main_network.global_step) < TOTAL_STEP:
        global_step = sess.run(main_network.global_step)
        if task_index == 0:
          # Synchronize the target network periodically (target network <- main network).
          if global_step >= next_target_network_update_step:
            sess.run(target_network.sync_op)
            next_target_network_update_step += TARGET_NETWORK_UPDATE_STEP
        
        # Sample action with epsilon-greedy policy.
        epsilon = EPSILON_MAX - (EPSILON_MAX - EPSILON_MIN) * np.minimum(global_step / EPSILON_DECAY_STEP, 1)
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
          list_episodic_reward.append((global_step, episodic_reward))
          delta_time = int(time.time() - start_time)
          print("Step ", global_step, "/", TOTAL_STEP, ": Time spent = ", delta_time, " s , Episodic reward = ", episodic_reward, sep = "")
          episodic_reward = 0
        # Store data.
        data = [obs, action, reward, done, obs_next]
        replay_buffer.append(data)
        # Update observation.
        obs = obs_next
        
        # Learning rate.
        lr = LEARNING_RATE[-1]
        for i in range(len(LR_ANNEAL_STEP)):
          if global_step < LR_ANNEAL_STEP[i]:
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
        
        # Update the main network (main network <- local network gradients).
        sess.run(main_network.train_op, feed_dict = {
          main_network.Obs: batch_obs / 255.0, main_network.Action: batch_action, main_network.TargetQ: target_q, main_network.LR: lr
          })
        
        if task_index == 0:
          # Save the main network periodically.
          if global_step >= next_autosave_step:
            saver.save(sess._sess._sess._sess._sess, SAVE_DIR + file_name)
            next_autosave_step += AUTOSAVE_STEP
      
      if task_index == 0:
        # Save the main network.
        saver.save(sess._sess._sess._sess._sess, SAVE_DIR + file_name)
    
    tf.contrib.keras.backend.clear_session()
    # Close the environment.
    env.close()
    
    queues = []
    # Create a shared queue on the worker which is visible on the parameter server.
    for i in range(cluster.num_tasks("ps")):
      with tf.device("/job:ps/task:" + str(i)):
        queue = tf.FIFOQueue(cluster.num_tasks("worker"), tf.int32, shared_name = "done_queue" + str(i))
        queues.append(queue)
    # Notify all parameter servers that the current worker has finished the task.
    with tf.Session(server.target) as sess:
      for i in range(cluster.num_tasks("ps")):
        sess.run(queues[i].enqueue(task_index))
    # Release memory when a worker is finished.
    tf.contrib.keras.backend.clear_session()
    
    return list_episodic_reward

def train(file_name):
  # Create folders.
  if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR)
  if not os.path.isdir(FIGURE_TRAINING_DIR):
    os.makedirs(FIGURE_TRAINING_DIR)
  
  start_time = time.time()
  
  # Create a cluster dictionary from the parameter server and workers.
  cluster_dict = {}
  cluster_dict.update({"ps": ["localhost:" + str(2220 + i) for i in range(NUM_PS)]})
  cluster_dict.update({"worker": ["localhost:" + str(2220 + NUM_PS + i) for i in range(NUM_WORKER)]})
  
  # Define the corresponding job and task.
  jobs = []
  for i in range(NUM_PS):
    jobs.append(("ps", i))
  for i in range(NUM_WORKER):
    jobs.append(("worker", i))
  
  # Multiprocessing.
  args = [(job[0], job[1], cluster_dict, file_name) for job in jobs]
  with multiprocessing.Pool(NUM_PS + NUM_WORKER) as p:
    lists_episodic_reward = p.starmap(worker_process, args)
  p.join()
  
  # Sort the list.
  list_episodic_reward = []
  for elem in lists_episodic_reward:
    list_episodic_reward.extend(elem)
  list_episodic_reward.sort()
  
  total_time = int(time.time() - start_time)
  print("Training finished in ", total_time, " s.", sep = "")
  
  # Plot the episodic reward against training step curve.
  plot_episodic_reward(list_episodic_reward, file_name)
  
if __name__ == "__main__":
  train(file_name = "async_dqn")