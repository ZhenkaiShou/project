import csv
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from config import *
from model import *

def plot_training_data(file_name):
  # Random seed for reproducible result.
  np.random.seed(0)
  
  # Create folders.
  if not os.path.isdir(FIGURE_DIR + "Average Loss/"):
    os.makedirs(FIGURE_DIR + "Average Loss/")
  if not os.path.isdir(FIGURE_DIR + "Training and Test Samples/"):
    os.makedirs(FIGURE_DIR + "Training and Test Samples/")
  
  # Generate training data.
  training_y = np.random.uniform(-10, 10, (TRAINING_SAMPLES, 1))
  training_x = 7 * np.sin(0.75 * training_y) + 0.5 * training_y + 1 * np.random.normal(-1, 1, (TRAINING_SAMPLES, 1))
  
  # Plot the training data.
  f, ax = plt.subplots(nrows=1, ncols=1, figsize = (5, 5))
  ax.plot(training_x, training_y, "b.", label = "Training")
  ax.set_ylim(-11, 11)
  ax.legend(loc = "lower right")
  
  f.savefig(FIGURE_DIR + "Training and Test Samples/" + file_name + ".png")
  plt.close(f)

def training(type, file_name):
  # Random seed for reproducible result.
  np.random.seed(0)
  tf.set_random_seed(54)
  
  # Create folders.
  if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR)
  if not os.path.isdir(CSV_DIR):
    os.makedirs(CSV_DIR)
  if not os.path.isdir(FIGURE_DIR):
    os.makedirs(FIGURE_DIR)
  if not os.path.isdir(FIGURE_DIR + "Average Loss/"):
    os.makedirs(FIGURE_DIR + "Average Loss/")
  if not os.path.isdir(FIGURE_DIR + "Training and Test Samples/"):
    os.makedirs(FIGURE_DIR + "Training and Test Samples/")
  
  if type == "mdn":
    mdn(file_name)
  else:
    fc(file_name)

def mdn(file_name):
  # Generate training data.
  training_y = np.random.uniform(-10, 10, (TRAINING_SAMPLES, 1))
  training_x = 7 * np.sin(0.75 * training_y) + 0.5 * training_y + 1 * np.random.normal(-1, 1, (TRAINING_SAMPLES, 1))
  # Generate test data (input only).
  test_y = np.random.uniform(-10, 10, (TEST_SAMPLES, 1))
  test_x = 7 * np.sin(0.75 * test_y) + 0.5 * test_y + 1 * np.random.normal(-1, 1, (TEST_SAMPLES, 1))
  
  model = MDN_Model()
  model.mdn()
  
  with tf.Session() as sess:
    # Initialize variables.
    sess.run(tf.global_variables_initializer())
    
    list_average_loss = []
    
    for epoch in range(EPOCH):
      # Determine learning rate based on the training steps.
      if epoch < ANNEALING_STEP[0]:
        lr = LEARNING_RATE[0]
      else:
        lr = LEARNING_RATE[1]
      
      # Training.
      [_, average_loss] = sess.run([model.train_op, model.total_loss], feed_dict = {model.X: training_x, model.Y: training_y, model.LR: lr})
      list_average_loss.append(average_loss)
      if epoch % 500 == 0 or epoch == EPOCH - 1:
        print("Epoch ",  epoch, ": average loss = ", average_loss, sep = '')
    
    # Save the parameters.
    saver = tf.train.Saver()
    saver.save(sess, SAVE_DIR + file_name)
    
    # Store data in the csv file.
    with open(CSV_DIR + file_name + ".csv", "w") as f:
      fieldnames = ["Epoch", "Average Loss"]
      writer = csv.DictWriter(f, fieldnames = fieldnames, lineterminator = "\n")
      writer.writeheader()
      for epoch in range(EPOCH):
        content = {"Epoch": epoch, "Average Loss": list_average_loss[epoch]}
        writer.writerow(content)
    
    # Create figure.
    list_epoch = list(range(EPOCH))
    
    f, ax = plt.subplots(nrows=1, ncols=1, figsize = (5, 5))
    ax.plot(list_epoch, list_average_loss)
    ax.set_title("Modes = " + str(MODES))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Loss")
    
    f.savefig(FIGURE_DIR + "Average Loss/" + file_name + ".png")
    plt.close(f)
    
    # Test.
    [logits, mu, sigma] = sess.run([model.logits, model.mu, model.sigma], feed_dict = {model.X: test_x})
    reduced_logits = logits - np.max(logits, -1, keepdims = True)
    pi = np.exp(reduced_logits / TEMPERATURE) / np.sum(np.exp(reduced_logits / TEMPERATURE), -1, keepdims = True)
    
    # Sample a mode from gaussian distribution pi.
    chosen_mode = np.reshape(np.array([np.random.choice(MODES, p = x) for x in pi]), [-1, 1])
    # Sample the output y from the corresponding mode.
    chosen_mu = np.reshape(np.array([mu[i, chosen_mode[i]] for i in range(TEST_SAMPLES)]), [-1, 1])
    chosen_sigma = np.reshape(np.array([sigma[i, chosen_mode[i]] for i in range(TEST_SAMPLES)]), [-1, 1])
    sample_y = chosen_mu + chosen_sigma * np.random.randn(TEST_SAMPLES, 1) * np.sqrt(TEMPERATURE)
    
    # Plot the training data and test data.
    f, ax = plt.subplots(nrows=1, ncols=1, figsize = (5, 5))
    ax.plot(training_x, training_y, "b.", label = "Training")
    ax.plot(test_x, sample_y, "r.", label = "Test")
    ax.set_ylim(-11, 11)
    ax.set_title("Modes = " + str(MODES) + ", Temperature = " + str(TEMPERATURE))
    ax.legend(loc = "lower right")
    
    f.savefig(FIGURE_DIR + "Training and Test Samples/" + file_name + ".png")
    plt.close(f)
  tf.contrib.keras.backend.clear_session()

def fc(file_name):
  # Generate training data.
  training_y = np.random.uniform(-10, 10, (TRAINING_SAMPLES, 1))
  training_x = 7 * np.sin(0.75 * training_y) + 0.5 * training_y + 1 * np.random.normal(-1, 1, (TRAINING_SAMPLES, 1))
  # Generate test data (input only).
  test_y = np.random.uniform(-10, 10, (TEST_SAMPLES, 1))
  test_x = 7 * np.sin(0.75 * test_y) + 0.5 * test_y + 1 * np.random.normal(-1, 1, (TEST_SAMPLES, 1))
  
  model = MDN_Model()
  model.fc()
  
  with tf.Session() as sess:
    # Initialize variables.
    sess.run(tf.global_variables_initializer())
    
    list_average_loss = []
    
    for epoch in range(EPOCH):
      # Determine learning rate based on the training steps.
      if epoch < ANNEALING_STEP[0]:
        lr = LEARNING_RATE[0]
      else:
        lr = LEARNING_RATE[1]
      
      # Training.
      [_, average_loss] = sess.run([model.train_op, model.total_loss], feed_dict = {model.X: training_x, model.Y: training_y, model.LR: lr})
      list_average_loss.append(average_loss)
      if epoch % 500 == 0 or epoch == EPOCH - 1:
        print("Epoch ",  epoch, ": average loss = ", average_loss, sep = '')
    
    # Save the parameters.
    saver = tf.train.Saver()
    saver.save(sess, SAVE_DIR + file_name)
    
    # Store data in the csv file.
    with open(CSV_DIR + file_name + ".csv", "w") as f:
      fieldnames = ["Epoch", "Average Loss"]
      writer = csv.DictWriter(f, fieldnames = fieldnames, lineterminator = "\n")
      writer.writeheader()
      for epoch in range(EPOCH):
        content = {"Epoch": epoch, "Average Loss": list_average_loss[epoch]}
        writer.writerow(content)
    
    # Create figure.
    list_epoch = list(range(EPOCH))
    
    f, ax = plt.subplots(nrows=1, ncols=1, figsize = (5, 5))
    ax.plot(list_epoch, list_average_loss)
    ax.set_title("Average Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Loss")
    
    f.savefig(FIGURE_DIR + "Average Loss/" + file_name + ".png")
    plt.close(f)
    
    # Test.
    predicted_y = sess.run(model.y, feed_dict = {model.X: test_x})
    
    # Plot the training data and test data.
    f, ax = plt.subplots(nrows=1, ncols=1, figsize = (5, 5))
    ax.plot(training_x, training_y, "b.", label = "Training")
    ax.plot(test_x, predicted_y, "r.", label = "Test")
    ax.set_ylim(-11, 11)
    ax.legend(loc = "lower right")
    
    f.savefig(FIGURE_DIR + "Training and Test Samples/" + file_name + ".png")
    plt.close(f)
  tf.contrib.keras.backend.clear_session()

def temperature_comparison(file_name, list_temperature):
  # Random seed for reproducible result.
  np.random.seed(0)
  tf.set_random_seed(54)
  
  # Create folders.
  if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR)
  if not os.path.isdir(FIGURE_DIR):
    os.makedirs(FIGURE_DIR)
  if not os.path.isdir(FIGURE_DIR + "Temperature Comparison/"):
    os.makedirs(FIGURE_DIR + "Temperature Comparison/")
  
  # Generate training data.
  training_y = np.random.uniform(-10, 10, (TRAINING_SAMPLES, 1))
  training_x = 7 * np.sin(0.75 * training_y) + 0.5 * training_y + 1 * np.random.normal(-1, 1, (TRAINING_SAMPLES, 1))
  # Generate test data (input only).
  test_y = np.random.uniform(-10, 10, (TEST_SAMPLES, 1))
  test_x = 7 * np.sin(0.75 * test_y) + 0.5 * test_y + 1 * np.random.normal(-1, 1, (TEST_SAMPLES, 1))
  
  model = MDN_Model()
  model.mdn()
  
  with tf.Session() as sess:
    # Load variables.
    saver = tf.train.Saver()
    saver.restore(sess, SAVE_DIR + file_name)
    
    # Compute the parameters for the test data.
    [logits, mu, sigma] = sess.run([model.logits, model.mu, model.sigma], feed_dict = {model.X: test_x})
    
    # Create gif for all each temperature in the list.
    imageio.mimsave(FIGURE_DIR + "Temperature Comparison/" + file_name + ".gif", 
    [plot_for_temperature(temperature, logits, mu, sigma, training_x, training_y, test_x) for temperature in list_temperature], fps = 1)

def plot_for_temperature(temperature, logits, mu, sigma, training_x, training_y, test_x):
  # Plot training and test samples for a single temperature.
  reduced_logits = logits - np.max(logits, -1, keepdims = True)
  pi = np.exp(reduced_logits / temperature) / np.sum(np.exp(reduced_logits / temperature), -1, keepdims = True)
  
  # Sample a mode from gaussian distribution pi.
  chosen_mode = np.reshape(np.array([np.random.choice(MODES, p = x) for x in pi]), [-1, 1])
  # Sample the output y from the corresponding mode.
  chosen_mu = np.reshape(np.array([mu[i, chosen_mode[i]] for i in range(TEST_SAMPLES)]), [-1, 1])
  chosen_sigma = np.reshape(np.array([sigma[i, chosen_mode[i]] for i in range(TEST_SAMPLES)]), [-1, 1])
  sample_y = chosen_mu + chosen_sigma * np.random.randn(TEST_SAMPLES, 1) * np.sqrt(temperature)
  
  # Plot the training data and test data.
  f, ax = plt.subplots(nrows=1, ncols=1, figsize = (5, 5))
  ax.plot(training_x, training_y, "b.", label = "Training")
  ax.plot(test_x, sample_y, "r.", label = "Test")
  ax.set_ylim(-11, 11)
  ax.set_title("Modes = " + str(MODES) + ", Temperature = " + str(temperature))
  ax.legend(loc = "lower right")
  
  # Used to return the plot as an image rray
  f.canvas.draw()
  image = np.frombuffer(f.canvas.tostring_rgb(), dtype = "uint8")
  image = image.reshape(f.canvas.get_width_height()[::-1] + (3,))
  
  return image

# Plot the training data.
plot_training_data(file_name = "training_data")

# Available parameters for type of function training(type, file_name)
# mdn: A mixture density network
# fc: A network with a single hidden layer
training(type = "mdn", file_name = "mdn_" + str(MODES))
training(type = "fc", file_name = "fc")

# file_name of function temperature_comparison(file_name, list_temperature)
# determines which file from "Saves" folder will be used to restore the network variables.
# temperature_comparison only works for MDNs.
temperature_comparison(file_name = "mdn_" + str(MODES), list_temperature = [0.1, 0.2, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0])