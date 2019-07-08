import matplotlib.pyplot as plt
import os
import pandas as pd

from config import *

def plot(input_file_names, output_file_name):
  # Create folders.
  if not os.path.isdir(FIGURE_TRAINING_DIR):
    os.makedirs(FIGURE_TRAINING_DIR)
  
  # Load all input files.
  records = []
  for input_file_name in input_file_names:
    record = pd.read_csv(CSV_DIR + input_file_name + ".csv", sep = ",")
    records.append(record)
  
  # Plot the training progress.
  f, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (5, 5))
  ax.set_title("Training Progress")
  ax.set_xlabel("Frame")
  ax.set_ylabel("Mean Episodic Reward")
  ax.grid()
  
  colors = "rbgcmyk"
  for i, (input_file_name, record) in enumerate(zip(input_file_names, records)):
    list_step = record["Step"].values
    list_mean_episodic_reward = record["Mean Episodic Reward"].values
    if i < len(colors):
      color = colors[i]
    else:
      color = np.random.uniform(size = (3,))
    ax.plot(list_step, list_mean_episodic_reward, color = color, linestyle = "-", label = input_file_name)
  ax.legend(loc = "lower right")
  
  f.savefig(FIGURE_TRAINING_DIR + output_file_name + ".png")
  plt.close(f)