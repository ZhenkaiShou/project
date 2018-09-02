import csv
import glob
import matplotlib.pyplot as plt
import numpy as np
import os.path
import tensorflow as tf
from tensorflow import keras
from config import *
from model import *

def training(type, file_name):
  # Random seed for reproducible result.
  np.random.seed(0)
  tf.set_random_seed(54)

  # Create folders.
  if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR)
  if not os.path.isdir(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)
  if not os.path.isdir(CSV_DIR):
    os.makedirs(CSV_DIR)
  if not os.path.isdir(FIGURE_DIR):
    os.makedirs(FIGURE_DIR)
  
  # Load data.
  mnist = tf.keras.datasets.mnist
  (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
  
  # Normalize and reshape training images.
  train_images = train_images / 255.0
  train_images = train_images.reshape(-1, 28, 28)
  # Convert label (range from 0 to 9) to one hot vector.
  train_labels = np.eye(10)[train_labels]
  train_length = np.shape(train_images)[0]
  
  # Normalize and reshape test images.
  test_images = test_images / 255.0
  test_images = test_images.reshape(-1, 28, 28)
  # Convert label (range from 0 to 9) to one hot vector.
  test_labels = np.eye(10)[test_labels]
  test_length = np.shape(test_images)[0]
  
  # Invoke the corresponding model.
  model = RNN_Model()
  if type == "rnn":
    model.rnn()
  elif type == "lstm":
    model.lstm()
  else:
    model.fc()
  # print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
  
  with tf.Session() as sess:
    if not os.path.isdir(SUMMARY_DIR + file_name + "/"):
      os.makedirs(SUMMARY_DIR + file_name + "/")
    old_summaries = glob.glob(SUMMARY_DIR + file_name + "/*")
    for f in old_summaries:
      os.remove(f)
    summary_writer = tf.summary.FileWriter(SUMMARY_DIR + file_name + "/", sess.graph)
    # Initialize variables.
    sess.run(tf.global_variables_initializer())
    
    list_average_loss = []
    list_training_accuracy = []
    list_test_accuracy = []
    
    for epoch in range(EPOCH):
      # Determine learning rate based on the training steps.
      if epoch < ANNEALING_STEP[0]:
        lr = LEARNING_RATE[0]
      else:
        lr = LEARNING_RATE[1]
      
      average_loss = 0
      training_accuracy = 0
      test_accuracy = 0
      
      # Training.
      for i in range(np.ceil(train_length / BATCH_SIZE).astype(int)):
        images = train_images[i*BATCH_SIZE:np.minimum((i+1)*BATCH_SIZE, train_length)]
        labels = train_labels[i*BATCH_SIZE:np.minimum((i+1)*BATCH_SIZE, train_length)]
        [_, total_loss, y] = sess.run([model.train_op, model.total_loss, model.y], feed_dict = {model.Inputs: images, model.Labels: labels, model.LR: lr})
        
        average_loss += total_loss
        comparison = np.equal(np.argmax(y, 1), np.argmax(labels, 1))
        correct = np.sum(comparison)
        training_accuracy += correct
      
      # Test.
      for i in range(np.ceil(test_length / BATCH_SIZE).astype(int)):
        images = test_images[i*BATCH_SIZE:np.minimum((i+1)*BATCH_SIZE, test_length)]
        labels = test_labels[i*BATCH_SIZE:np.minimum((i+1)*BATCH_SIZE, test_length)]
        y = sess.run(model.y, feed_dict = {model.Inputs: images, model.Labels: labels, model.LR: lr})
        
        comparison = np.equal(np.argmax(y, 1), np.argmax(labels, 1))
        correct = np.sum(comparison)
        test_accuracy += correct
      
      average_loss /= np.ceil(train_length / BATCH_SIZE)
      training_accuracy /= train_length
      test_accuracy /= test_length
      
      list_average_loss.append(average_loss)
      list_training_accuracy.append(training_accuracy)
      list_test_accuracy.append(test_accuracy)
      
      # Tensorboard.
      summary=tf.Summary()
      summary.value.add(tag = "Average Loss", simple_value = average_loss)
      summary.value.add(tag = "Training Accuracy", simple_value = training_accuracy)
      summary.value.add(tag = "Test Accuracy", simple_value = test_accuracy)
      summary_writer.add_summary(summary, epoch)
      summary_writer.flush()
      
      print("Epoch ", format(epoch, "03d"), ": average_loss = ", format(average_loss, ".8f"), ", training accuracy = ", format(training_accuracy, ".2%"), ", test accuracy = ", format(test_accuracy, ".2%"), sep = '')
    
    # Save the parameters.
    saver = tf.train.Saver()
    saver.save(sess, SAVE_DIR + file_name)
    
    # Store data in the csv file.
    with open(CSV_DIR + file_name + ".csv", "w") as f:
      fieldnames = ["Epoch", "Average Loss", "Training Accuracy", "Test Accuracy"]
      writer = csv.DictWriter(f, fieldnames = fieldnames, lineterminator = "\n")
      writer.writeheader()
      for epoch in range(EPOCH):
        content = {"Epoch": epoch, "Average Loss": list_average_loss[epoch], "Training Accuracy": list_training_accuracy[epoch], "Test Accuracy": list_test_accuracy[epoch]}
        writer.writerow(content)
    
    # Create figure.
    list_epoch = list(range(EPOCH))
    f, axes = plt.subplots(nrows=1, ncols=3, figsize = (15, 5))
    
    axes[0].plot(list_epoch, list_average_loss)
    axes[0].set_title("Average Loss")
    
    axes[1].plot(list_epoch, list_training_accuracy)
    axes[1].set_title("Training Accuracy")
    axes[1].set_ylim([0.6, 1.0])
    
    axes[2].plot(list_epoch, list_test_accuracy)
    axes[2].set_title("Test Accuracy")
    axes[2].set_ylim([0.6, 1.0])

    f.savefig(FIGURE_DIR + file_name + ".png")
    plt.close(f)
  tf.contrib.keras.backend.clear_session()

# Available parameters for type of function training(type, file_name)
# rnn: A basic recurrent neural network
# lstm: A Long Short-Term Memory Network
# fc: A network with a single hidden layer
training(type = "rnn", file_name = "rnn")
training(type = "lstm", file_name = "lstm")
training(type = "fc", file_name = "fc")