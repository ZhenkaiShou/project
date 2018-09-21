import cma
import csv
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os

import config
from controller import Controller
from env import MyCarRacing
from rnn_mdn import RNN_MDN
from vae import VAE

# Local hyperparameters
TARGET_SCORE = 900.0
EVAL_TRIAL = 100
EVAL_INTERVAL = 10
POPSIZE = 64
SIGMA0 = 0.1
PROCESS = 2
EPISODE_PER_SOLUTION = 8
CONTROLLER_MODE = config.CONTROLLER_MODE
MAX_FRAME = config.MAX_FRAME
RENDER_MODE = config.RENDER_MODE

USING_GPU = False
MEMORY = 0.1

SAVE_VAE_DIR = config.SAVE_VAE_DIR
SAVE_RNN_DIR = config.SAVE_RNN_DIR
SAVE_CONTROLLER_DIR = config.SAVE_CONTROLLER_DIR
CSV_DIR = config.CSV_DIR
FIGURE_TRAINING_DIR = config.FIGURE_TRAINING_DIR

def controller_training(vae_file_name = "vae", rnn_file_name = "rnn", file_name = "controller"):
  # Create folders.
  if not os.path.isdir(SAVE_CONTROLLER_DIR):
    os.makedirs(SAVE_CONTROLLER_DIR)
  if not os.path.isdir(CSV_DIR):
    os.makedirs(CSV_DIR)
  if not os.path.isdir(FIGURE_TRAINING_DIR):
    os.makedirs(FIGURE_TRAINING_DIR)
  
  # Determine parameter size.
  con = Controller()
  params_size = np.prod(np.shape(con.weights)) + np.prod(np.shape(con.bias))
  
  # CMA-ES.
  es = cma.CMAEvolutionStrategy(np.zeros(params_size), SIGMA0, {"popsize": POPSIZE})
  
  list_training_reward = []
  list_eval_reward = []
  step = 0
  
  while True:
    # Sample solutions from N(mu, sigma^2).
    solutions = es.ask()
    
    # Feed solutions to each process.
    start_index = 0
    base = POPSIZE // PROCESS
    reminder = POPSIZE % PROCESS
    arg_list = [None for _ in range(PROCESS)]
    for i in range(PROCESS):
      num_solution = base + 1 if i < reminder else base
      list_solution = solutions[start_index:start_index + num_solution]
      list_episode = [EPISODE_PER_SOLUTION] * num_solution
      arg_list[i] = [list_solution, list_episode, vae_file_name, rnn_file_name]
      start_index += num_solution
    
    with multiprocessing.Pool(PROCESS) as p:
      # Shape of list_total_reward = [PROCESS, num_solution for each process, num_episode for each solution]
      list_total_reward = p.starmap(run_env_process, arg_list)
    list_total_reward = [np.mean(item) for sublist in list_total_reward for item in sublist]
    # Fitness value is the negative average reward.
    list_fitness_value = [-value for value in list_total_reward]
    best_index = np.argmax(list_total_reward)
    best_solution = solutions[best_index]
    highest_reward = list_total_reward[best_index]
    
    # Update solutions.
    es.tell(solutions, list_fitness_value)
    list_training_reward.append(highest_reward)
    
    print("Step ", format(step, "04d"), ":", sep = "") 
    print("  Training Reward = ", format(highest_reward, ".8f"), sep = "")
    
    if step % EVAL_INTERVAL == 0:
      # Evaluate the best solution over 100 random trials.
      base = EVAL_TRIAL // PROCESS
      reminder = EVAL_TRIAL % PROCESS
      arg_list = [None for _ in range(PROCESS)]
      for i in range(PROCESS):
        num_episode = base + 1 if i < reminder else base
        arg_list[i] = [[best_solution], [num_episode], vae_file_name, rnn_file_name]
      
      with multiprocessing.Pool(PROCESS) as p:
        # Shape of list_total_reward = [PROCESS, 1, num_episode]
        list_total_reward = p.starmap(run_env_process, arg_list)
      list_total_reward = [item for sublist in list_total_reward for subsublist in sublist for item in subsublist]
      # Compute the average reward of the 100 random trials.
      eval_reward = np.mean(list_total_reward)
      list_eval_reward.append(eval_reward)
      print("  Evaluation Reward = ", format(eval_reward, ".8f"), sep = "")
      
      # Save file.
      weights = np.reshape(best_solution[:np.prod(np.shape(con.weights))], np.shape(con.weights))
      bias = np.reshape(best_solution[-np.prod(np.shape(con.bias)):], np.shape(con.bias))
      np.savez_compressed(SAVE_CONTROLLER_DIR + file_name, weights = weights, bias = bias)
      
      num_step = len(list_training_reward)
      # Store data in the csv file.
      with open(CSV_DIR + file_name + ".csv", "w") as f:
        fieldnames = ["Step", "Training Reward", "Eval Reward"]
        writer = csv.DictWriter(f, fieldnames = fieldnames, lineterminator = "\n")
        writer.writeheader()
        for s in range(num_step):
          if s % EVAL_INTERVAL == 0:
            content = {"Step": s, "Training Reward": list_training_reward[s], "Eval Reward": list_eval_reward[s // EVAL_INTERVAL]}
          else:
            content = {"Step": s, "Training Reward": list_training_reward[s]}
          writer.writerow(content)
      
      # Plot the training loss.
      list_step = list(range(num_step))
      f, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (5, 5))
      ax.plot(list_step, list_training_reward, "r-", label = "Training Reward")
      ax.plot(list_step[::EVAL_INTERVAL], list_eval_reward, "b-", label = "Evaluation Reward")
      ax.set_title("Performance")
      ax.set_xlabel("Step")
      ax.set_ylabel("Cumulative Reward")
      ax.legend(loc = "lower right")
      ax.grid()
      
      f.savefig(FIGURE_TRAINING_DIR + file_name + ".png")
      plt.close(f)
      
      # Stop evaluation if the condition is met
      if eval_reward >= TARGET_SCORE:
        break
    
    # Update the step.
    step += 1

def run_env_process(list_solution, list_episode, vae_file_name, rnn_file_name):
  import tensorflow as tf
  
  if USING_GPU:
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = MEMORY)
    gpu_config = tf.ConfigProto(gpu_options = gpu_options)
  else:
    gpu_config = None
  
  np.random.seed()
  
  # Load models.
  vae = VAE()
  vae.build_model(is_training = False, is_assigning = True)
  rnn = RNN_MDN()
  rnn.build_model(is_training = False, is_assigning = True, is_single_input = True)
  con = Controller()
  
  sess_vae = tf.Session(graph = vae.graph, config = gpu_config)
  sess_rnn = tf.Session(graph = rnn.graph, config = gpu_config)
  
  # Load VAE and RNN variables.
  with vae.graph.as_default():
    saver_vae = tf.train.Saver()
    saver_vae.restore(sess_vae, SAVE_VAE_DIR + vae_file_name)
  with rnn.graph.as_default():
    saver_rnn = tf.train.Saver()
    saver_rnn.restore(sess_rnn, SAVE_RNN_DIR + rnn_file_name)
  
  env = MyCarRacing()
  list_total_reward = [[] for _ in range(len(list_solution))]
  
  for i in range(len(list_solution)):
    # Load controller variables.
    con.weights = np.reshape(list_solution[i][:np.prod(np.shape(con.weights))], np.shape(con.weights))
    con.bias = np.reshape(list_solution[i][-np.prod(np.shape(con.bias)):], np.shape(con.bias))
    
    for j in range(list_episode[i]):
      # Get intitial state of RNN.
      state = sess_rnn.run(rnn.initial_state)
      
      # Reset the environment.
      obs = env.reset()
      total_reward = 0
      
      for step in range(MAX_FRAME):
        env.render(mode = RENDER_MODE)
        
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
        
        # Update the hidden state.
        za = np.reshape(np.concatenate((z, action), 1), (1, 1, -1))
        state = sess_rnn.run(rnn.final_state, feed_dict = {rnn.ZA: za, rnn.initial_state: state})
        
        # Interact with the game engine.
        obs, reward, done, _ = env.step(action[0])
        total_reward += reward
        
        # Early stop if the game is finished.
        if done:
          break
      
      # Record the total reward.
      list_total_reward[i].append(total_reward)
  
  env.render(close = True)
  sess_vae.close()
  sess_rnn.close()
  tf.contrib.keras.backend.clear_session()
  
  return list_total_reward

if __name__ == "__main__":
  controller_training()