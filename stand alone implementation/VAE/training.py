import csv
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from config import *
from model import *

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
  if not os.path.isdir(FIGURE_DIR + "Reconstruction/"):
    os.makedirs(FIGURE_DIR + "Reconstruction/")
  if not os.path.isdir(FIGURE_DIR + "Latent Representation/"):
    os.makedirs(FIGURE_DIR + "Latent Representation/")
  
  # Load data.
  mnist = tf.keras.datasets.mnist
  (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
  
  # Normalize and reshape training images.
  training_images = training_images / 255.0
  training_images = np.reshape(training_images, [-1, INPUT_LENGTH * INPUT_WIDTH])
  training_length = np.shape(training_images)[0]
  
  # Normalize and reshape test images.
  test_images = test_images / 255.0
  test_images = np.reshape(test_images, [-1, INPUT_LENGTH * INPUT_WIDTH])
  test_length = np.shape(test_images)[0]
  
  # Invoke the corresponding model.
  model = VAE_Model()
  if type == "vae":
    model.vae()
  else:
    model.ae()
  
  with tf.Session() as sess:
    # Initialize variables.
    sess.run(tf.global_variables_initializer())
    
    list_training_loss = []
    list_test_loss = []
    
    for epoch in range(EPOCH):
      # Shuffle the training data.
      random_index = np.array(range(training_length))
      np.random.shuffle(random_index)
      random_training_images = training_images[random_index]
      
      training_loss = 0
      test_loss = 0
      
      # Training.
      for i in range(np.ceil(training_length / BATCH_SIZE).astype(int)):
        images = random_training_images[i*BATCH_SIZE:np.minimum((i+1)*BATCH_SIZE, training_length)]
        [_, total_loss] = sess.run([model.train_op, model.total_loss], feed_dict = {model.Inputs: images})
        training_loss += total_loss * np.shape(images)[0]
      
      # Validation.
      for i in range(np.ceil(test_length / BATCH_SIZE).astype(int)):
        images = test_images[i*BATCH_SIZE:np.minimum((i+1)*BATCH_SIZE, test_length)]
        total_loss = sess.run(model.total_loss, feed_dict = {model.Inputs: images})
        test_loss += total_loss * np.shape(images)[0]
      
      training_loss /= training_length
      test_loss /= test_length
      list_training_loss.append(training_loss)
      list_test_loss.append(test_loss)
      
      print("Epoch ", format(epoch, "03d"), ": Training Loss = ", format(training_loss, ".8f"), ", Test Loss = ", format(test_loss, ".8f"), sep = '')
    
    # Save the parameters.
    saver = tf.train.Saver()
    saver.save(sess, SAVE_DIR + file_name)
    
    # Store data in the csv file.
    with open(CSV_DIR + file_name + ".csv", "w") as f:
      fieldnames = ["Epoch", "Training Loss", "Test Loss"]
      writer = csv.DictWriter(f, fieldnames = fieldnames, lineterminator = "\n")
      writer.writeheader()
      for epoch in range(EPOCH):
        content = {"Epoch": epoch, "Training Loss": list_training_loss[epoch], "Test Loss": list_test_loss[epoch]}
        writer.writerow(content)
    
    # Plot the average loss.
    list_epoch = list(range(EPOCH))
    
    f, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (5, 5))
    ax.plot(list_epoch, list_training_loss, "r-", label = "Training")
    ax.plot(list_epoch, list_test_loss, "b-", label = "Test")
    ax.set_title("Average Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc = "upper right")
    ax.grid()
    
    f.savefig(FIGURE_DIR + "Average Loss/" + file_name + ".png")
    plt.close(f)
    
    # Test.
    [z, outputs] = sess.run([model.z, model.outputs], feed_dict = {model.Inputs: test_images})
    
    # Plot reconstruction images.
    list_index = [np.where(np.equal(test_labels, i))[0] for i in range(N_CLASS)]
    random_index = [np.random.choice(list_index[i], 2, replace = False) for i in range(N_CLASS)]
    random_index = np.transpose(random_index, (1, 0))
    random_index = np.reshape(random_index, (-1,))
    imageio.mimsave(FIGURE_DIR + "Reconstruction/" + file_name + ".gif", 
    [plot_reconstruction_image(test_images[index], outputs[index]) for index in random_index], fps = 1)
    
    # Plot the latent representation.
    f, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (6, 5))
    ax.set_title("Latent Representation")
    cmap = plt.get_cmap("jet", N_CLASS)
    scatter = ax.scatter(z[:, 0], z[:, 1], c = test_labels, cmap = cmap)
    cbar = plt.colorbar(scatter)
    loc = (np.arange(0, N_CLASS) + 0.5) * (N_CLASS - 1) / N_CLASS
    cbar.set_ticks(loc)
    cbar.ax.set_yticklabels(np.arange(0, N_CLASS))
    cbar.set_label("Class Label")
    ax.grid()
    
    f.savefig(FIGURE_DIR + "Latent Representation/" + file_name + ".png")
    plt.close(f)
  tf.contrib.keras.backend.clear_session()

def plot_reconstruction_image(input, output):
  # Plot both the original image and the reconstruction.
  f, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (6, 3))
  ax[0].imshow(np.reshape(input, (INPUT_LENGTH, INPUT_WIDTH)), cmap = "gray")
  ax[0].set_title("Input")
  ax[1].imshow(np.reshape(output, (INPUT_LENGTH, INPUT_WIDTH)), cmap = "gray")
  ax[1].set_title("Reconstruction")
  f.tight_layout()
  
  f.canvas.draw()
  image = np.frombuffer(f.canvas.tostring_rgb(), dtype = "uint8")
  image = image.reshape(f.canvas.get_width_height()[::-1] + (3,))
  plt.close(f)
  
  return image

def plot_mixture_reconstructions(type, file_name, n1, n2):
  # Random seed for reproducible result.
  np.random.seed(15)
  tf.set_random_seed(54)
  
  # Create folders.
  if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR)
  if not os.path.isdir(FIGURE_DIR):
    os.makedirs(FIGURE_DIR)
  if not os.path.isdir(FIGURE_DIR + "Mixture Reconstruction/"):
    os.makedirs(FIGURE_DIR + "Mixture Reconstruction/")
  
  # Load data.
  mnist = tf.keras.datasets.mnist
  (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
  
  # Normalize and reshape test images.
  test_images = test_images / 255.0
  test_images = np.reshape(test_images, [-1, INPUT_LENGTH * INPUT_WIDTH])
  test_length = np.shape(test_images)[0]
  
  # Randomly sample 2 test images with label n1 and n2.
  list_index = [np.where(np.equal(test_labels, n))[0] for n in [n1, n2]]
  random_index = [np.random.choice(list_index[i]) for i in range(len(list_index))]
  random_index = np.reshape(random_index, (-1,))
  random_images = test_images[random_index]
  random_labels = test_labels[random_index]
    
  # Invoke the corresponding model.
  model = VAE_Model()
  if type == "vae":
    model.vae()
  else:
    model.ae()
  
  with tf.Session() as sess:
    # Load variables.
    saver = tf.train.Saver()
    saver.restore(sess, SAVE_DIR + file_name)
    
    # Compute the reconstruction image of a merged latent representation.
    [z, outputs] = sess.run([model.z, model.outputs], feed_dict = {model.Inputs: random_images})
    parts = 20
    merged_z = np.array([np.average(z, 0, weights = [1 - 1.0 / parts * i, 1.0 / parts * i]) for i in range(parts + 1)])
    merged_output = sess.run(model.outputs, feed_dict = {model.z: merged_z})
  tf.contrib.keras.backend.clear_session()
  
  imageio.mimsave(FIGURE_DIR + "Mixture Reconstruction/" + file_name + "_" + str(n1) + "_" + str(n2) + ".gif", 
  [plot_mixture_reconstruction(random_images, z, merged_z[i], merged_output[i]) for i in range(parts + 1)], fps = 2)
  
def plot_mixture_reconstruction(inputs, z, merged_z, merged_output):
  # Plot the reconstruction.
  f, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (6, 6))
  ax[0, 0].imshow(np.reshape(inputs[0], (INPUT_LENGTH, INPUT_WIDTH)), cmap = "gray")
  ax[0, 0].set_title("First Input", color = "r")
  ax[0, 1].imshow(np.reshape(inputs[1], (INPUT_LENGTH, INPUT_WIDTH)), cmap = "gray")
  ax[0, 1].set_title("Second Input", color = "b")
  ax[1, 0].plot(z[0, 0], z[0, 1], "r.", label = "First Input")
  ax[1, 0].plot(z[1, 0], z[1, 1], "b.", label = "Second Input")
  ax[1, 0].plot(merged_z[0], merged_z[1], "k.", label = "Mean Value")
  ax[1, 0].set_title("Latent Representation")
  ax[1, 0].grid()
  ax[1, 1].imshow(np.reshape(merged_output, (INPUT_LENGTH, INPUT_WIDTH)), cmap = "gray")
  ax[1, 1].set_title("Reconstruction")
  f.tight_layout()
  
  f.canvas.draw()
  image = np.frombuffer(f.canvas.tostring_rgb(), dtype = "uint8")
  image = image.reshape(f.canvas.get_width_height()[::-1] + (3,))
  plt.close(f)

  return image

def plot_mixture_reconstructions_2(type, file_name, z1, z2):
  # Random seed for reproducible result.
  np.random.seed(15)
  tf.set_random_seed(54)
  
  # Create folders.
  if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR)
  if not os.path.isdir(FIGURE_DIR):
    os.makedirs(FIGURE_DIR)
  if not os.path.isdir(FIGURE_DIR + "Mixture Reconstruction 2/"):
    os.makedirs(FIGURE_DIR + "Mixture Reconstruction 2/")
  
  # Load data.
  mnist = tf.keras.datasets.mnist
  (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
  
  # Normalize and reshape test images.
  test_images = test_images / 255.0
  test_images = np.reshape(test_images, [-1, INPUT_LENGTH * INPUT_WIDTH])
  test_length = np.shape(test_images)[0]
    
  # Invoke the corresponding model.
  model = VAE_Model()
  if type == "vae":
    model.vae()
  else:
    model.ae()
  
  with tf.Session() as sess:
    # Load variables.
    saver = tf.train.Saver()
    saver.restore(sess, SAVE_DIR + file_name)
    
    # Compute the latent representation of all test images.
    z_test = sess.run(model.z, feed_dict = {model.Inputs: test_images})
    
    # Compute the reconstruction image of a merged latent representation.
    z_input = np.array([z1, z2])
    parts = 50
    merged_z = np.array([np.average(z_input, 0, weights = [1 - 1.0 / parts * i, 1.0 / parts * i]) for i in range(parts + 1)])
    merged_output = sess.run(model.outputs, feed_dict = {model.z: merged_z})
    output_endpoints = np.array([merged_output[0], merged_output[-1]])
  tf.contrib.keras.backend.clear_session()
  
  imageio.mimsave(FIGURE_DIR + "Mixture Reconstruction 2/" + file_name + ".gif", 
  [plot_mixture_reconstruction_2(output_endpoints, z_test, test_labels, z_input, merged_z[i], merged_output[i]) for i in range(parts + 1)], fps = 3)
  
def plot_mixture_reconstruction_2(output_endpoints, z_test, test_labels, z_input, merged_z, merged_output):
  # Plot the reconstruction.
  f, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (6, 6))
  ax[0, 0].imshow(np.reshape(output_endpoints[0], (INPUT_LENGTH, INPUT_WIDTH)), cmap = "gray")
  ax[0, 0].set_title("Start", color = "r")
  ax[0, 1].imshow(np.reshape(output_endpoints[1], (INPUT_LENGTH, INPUT_WIDTH)), cmap = "gray")
  ax[0, 1].set_title("End", color = "b")
  ax[1, 0].plot(z_input[0, 0], z_input[0, 1], "r.", label = "Start")
  ax[1, 0].plot(z_input[1, 0], z_input[1, 1], "b.", label = "End")
  ax[1, 0].plot(merged_z[0], merged_z[1], "k.", label = "Linear Interpolation")
  ax[1, 0].set_title("Latent Representation")
  ax[1, 0].grid()
  cmap = plt.get_cmap("jet", N_CLASS)
  scatter = ax[1, 0].scatter(z_test[:, 0], z_test[:, 1], c = test_labels, cmap = cmap)
  #cbar = plt.colorbar(scatter)
  #loc = (np.arange(0, N_CLASS) + 0.5) * (N_CLASS - 1) / N_CLASS
  #cbar.set_ticks(loc)
  #cbar.ax.set_yticklabels(np.arange(0, N_CLASS))
  # cbar.set_label("Class Label")  
  ax[1, 1].imshow(np.reshape(merged_output, (INPUT_LENGTH, INPUT_WIDTH)), cmap = "gray")
  ax[1, 1].set_title("Reconstruction")
  f.tight_layout()
  
  f.canvas.draw()
  image = np.frombuffer(f.canvas.tostring_rgb(), dtype = "uint8")
  image = image.reshape(f.canvas.get_width_height()[::-1] + (3,))
  plt.close(f)

  return image

# Function training(type, file_name):
#
# Available parameters for type:
# - "vae": A variational autoencoder
# - "ae": An autoencoder
training(type = "vae", file_name = "vae_" + str(LATENT_UNITS))
training(type = "ae", file_name = "ae_" + str(LATENT_UNITS))

# Function plot_mixture_reconstruction(type, file_name, n1, n2)
#
# Available parameters for type:
# - "vae": A variational autoencoder
# - "ae": An autoencoder
# file_name determines which file from "Saves" folder will be used to restore the network variables.
# n1, n2 determines the class label of two randomly sampled images.
# - available parameters: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
plot_mixture_reconstructions(type = "vae", file_name = "vae_" + str(LATENT_UNITS), n1 = 2, n2 = 7)

# Function plot_mixture_reconstruction_2(type, file_name, z1, z2)
#
# Available parameters for type:
# - "vae": A variational autoencoder
# - "ae": An autoencoder
# file_name determines which file from "Saves" folder will be used to restore the network variables.
# z1, z2 determines the starting and ending position in the latent space.
plot_mixture_reconstructions_2(type = "vae", file_name = "vae_" + str(LATENT_UNITS), z1 = np.array([1, 3]), z2 = np.array([4, -1]))
