import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import time

from config import *
from model import Model

def plot(input_file_names, output_file_name):
  # Create folders.
  if not os.path.isdir(FIGURE_DIR):
    os.makedirs(FIGURE_DIR)
  
  # Load all input files.
  records = []
  for input_file_name in input_file_names:
    record = pd.read_csv(RECORD_DIR + input_file_name + ".csv", sep = ",")
    records.append(record)
  
  # Plot the training progress.
  f, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (11, 5))
  ax[0].set_title("Average Loss")
  ax[0].set_xlabel("Epoch")
  ax[0].set_ylabel("Loss")
  ax[0].set_ylim(0.0, 3.0)
  ax[0].grid()
  ax[1].set_title("Average Error")
  ax[1].set_xlabel("Epoch")
  ax[1].set_ylabel("Error")
  ax[1].set_ylim(0.0, 0.6)
  ax[1].grid()
  
  colors = "rbgcmyk"
  for i, (input_file_name, record) in enumerate(zip(input_file_names, records)):
    list_epoch = record["Epoch"].values
    list_train_loss = record["Training Loss"].values
    list_train_error = record["Training Error"].values
    list_test_loss = record["Test Loss"].values
    list_test_error = record["Test Error"].values
    if i < len(colors):
      color = colors[i]
    else:
      color = np.random.uniform(size = (3,))
    ax[0].plot(list_epoch, list_train_loss, color = color, linestyle = "-")
    ax[0].plot(list_epoch, list_test_loss, color = color, linestyle = "-", linewidth = 3, label = input_file_name)
    ax[1].plot(list_epoch, list_train_error, color = color, linestyle = "-")
    ax[1].plot(list_epoch, list_test_error, color = color, linestyle = "-", linewidth = 3, label = input_file_name)
  ax[0].legend(loc = "upper right")
  ax[1].legend(loc = "upper right")

  f.savefig(FIGURE_DIR + output_file_name + ".png")
  plt.close(f)

def train(file_name,
  epoch = 120,
  batch_size = 100,
  
  optimizer = "Adam",
  learning_rate = [1e-3, 1e-4, 1e-5],
  lr_schedule = [60, 90],
  
  normalize_data = False,
  flip_data = False,
  crop_data = False,
  
  network_type = "Res4",
  dropout_rate = 0.2,
  c_l2 = 0.0,
  batch_norm = True,
  global_average_pool = True
  ):
  # Hyperparameters.
  hps = {
    "epoch": epoch,
    "batch_size": batch_size,
    
    "optimizer": optimizer,
    "learning_rate": learning_rate,
    "lr_schedule": lr_schedule,
    
    "normalize_data": normalize_data,
    "flip_data": flip_data,
    "crop_data": crop_data,
    
    "network_type": network_type,
    "dropout_rate": dropout_rate,
    "c_l2": c_l2,
    "batch_norm": batch_norm,
    "global_average_pool": global_average_pool
  }
  
  # Create folders.
  if not os.path.isdir(RECORD_DIR):
    os.makedirs(RECORD_DIR)
  if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR)
  
  # Load data.
  (data_train, label_train), (data_test, label_test) = tf.keras.datasets.cifar10.load_data()
  data_train = data_train / 255.0
  data_test = data_test / 255.0
  
  if hps["normalize_data"]:
    # Normalize data.
    data_train_mean = np.mean(data_train, axis = 0)
    data_train -= data_train_mean
    data_test -= data_train_mean
  
  # Compute data length.
  train_length = data_train.shape[0]
  test_length = data_test.shape[0]
  num_train_batch = train_length // hps["batch_size"]
  num_test_batch = int(np.ceil(test_length / hps["batch_size"]))
  
  # Build models.
  model = Model(hps)
  variables_initializer = tf.global_variables_initializer()
  
  with tf.Session() as sess:
    sess.run(variables_initializer)
    
    t0 = time.time()
    list_train_loss = []
    list_train_error = []
    list_test_loss = []
    list_test_error = []
    
    for epoch in range(hps["epoch"]):
      # Shuffle training data.
      random_id = np.arange(train_length)
      np.random.shuffle(random_id)
      
      # Learning rate.
      lr = hps["learning_rate"][-1]
      for i in range(len(hps["lr_schedule"])):
        if epoch < hps["lr_schedule"][i]:
          lr = hps["learning_rate"][i]
          break
      
      # Training phase.
      avg_train_loss = 0
      avg_train_error = 0
      for batch in range(num_train_batch):
        batch_id = random_id[batch * hps["batch_size"] : (batch+1) * hps["batch_size"]]
        batch_data = data_train[batch_id]
        batch_label = label_train[batch_id]
        batch_label = np.reshape(batch_label, -1)
        _, loss, wrong_pred = sess.run([model.train_op, model.loss, model.wrong_pred], 
          feed_dict = {model.Input: batch_data, model.Label: batch_label, model.LR: lr, model.IsTraining: True})
        avg_train_loss += loss
        avg_train_error += wrong_pred
      avg_train_loss /= num_train_batch
      avg_train_error /= (num_train_batch * hps["batch_size"])
      list_train_loss.append(avg_train_loss)
      list_train_error.append(avg_train_error)
      
      # Test phase.
      avg_test_loss = 0
      avg_test_error = 0
      for batch in range(num_test_batch):
        batch_id = np.arange(batch * hps["batch_size"], np.minimum((batch+1) * hps["batch_size"], test_length))
        batch_data = data_test[batch_id]
        batch_label = label_test[batch_id]
        batch_label = np.reshape(batch_label, -1)
        loss, wrong_pred = sess.run([model.loss, model.wrong_pred], 
          feed_dict = {model.Input: batch_data, model.Label: batch_label, model.IsTraining: False})
        avg_test_loss += loss * batch_id.shape[0]
        avg_test_error += wrong_pred
      avg_test_loss /= test_length
      avg_test_error /= test_length
      list_test_loss.append(avg_test_loss)
      list_test_error.append(avg_test_error)
      print("Epoch ", format(epoch, "03d"), ": Training Loss = ", format(avg_train_loss, ".6f"), ", Training Error = ", format(avg_train_error, ".4f"), 
        ", Test Loss = ", format(avg_test_loss, ".6f"), ", Test Error = ", format(avg_test_error, ".4f"), sep = "")
    
    delta_t = time.time() - t0
    print("Training finished in ", format(delta_t, ".2f"), " s.", sep = "")
    
    # Save parameters.
    saver = tf.train.Saver()
    saver.save(sess, SAVE_DIR + file_name)
    
    # Store record into a csv file.
    list_epoch = list(range(hps["epoch"]))
    record = pd.DataFrame({"Epoch": list_epoch, "Training Loss": list_train_loss, "Training Error": list_train_error,
      "Test Loss": list_test_loss, "Test Error": list_test_error})
    record.to_csv(RECORD_DIR + file_name + ".csv", sep = ",", index = False)
  tf.contrib.keras.backend.clear_session()  
  
if __name__ == "__main__":
  '''
  Default hyperparameter settings in train().
  ============================================================================================== 
  epoch = 120                        # Number of epochs
  batch_size = 100                   # Minibatch size
  
  optimizer = "Adam"                 # Available optimizer, choose between ("Momentum" | "Adam")
  learning_rate = [1e-3, 1e-4, 1e-5] # Learning rate for each phase
  lr_schedule = [60, 90]             # Epochs required to reach the next learning rate phase
  
  normalize_data = False             # Whether input images are normalized
  flip_data = False                  # Whether input images are flipped with 50% chance
  crop_data = False                  # Whether input images are zero-padded and randomly cropped
  
  network_type = "Res4"              # Network type, choose between ("Res4" | "Conv8" | "None")
  dropout_rate = 0.2                 # Dropout rate, value of 0 means no dropout
  c_l2 = 0.0                         # L2 regularization, also known as weight decay
  batch_norm = True                  # Whether batch normalization is applied
  global_average_pool = True         # Whether global average pooling is applied
  ==============================================================================================
  '''
  train(file_name = "res4")
  train(file_name = "conv8", network_type = "Conv8")
  train(file_name = "simple network", network_type = "None")
  train(file_name = "res4, no dropout", dropout_rate = 0.0)
  train(file_name = "res4, L2", c_l2 = 1e-4)
  train(file_name = "res4, L2, no dropout", dropout_rate = 0.0, c_l2 = 1e-4)
  train(file_name = "res4, no batch norm", batch_norm = False)
  train(file_name = "res4, no global pool", global_average_pool = False)
  train(file_name = "res4, normalize data", normalize_data = True)
  train(file_name = "res4, flip data", flip_data = True)
  train(file_name = "res4, crop data", crop_data = True)
  train(file_name = "res4, augment data", flip_data = True, crop_data = True)
  train(file_name = "res4, momentum optimizer", optimizer = "Momentum", learning_rate = [1e-1, 1e-2, 1e-3])
  
  plot(input_file_names = ["res4"], output_file_name = "Res4")
  plot(input_file_names = ["res4", "conv8", "simple network"], output_file_name = "Network Type Comparison")
  plot(input_file_names = ["res4", "res4, no dropout", "res4, L2", "res4, L2, no dropout"], output_file_name = "Regularization Comparison")
  plot(input_file_names = ["res4", "res4, no batch norm"], output_file_name = "Batch Norm Comparison")
  plot(input_file_names = ["res4", "res4, no global pool"], output_file_name = "Global Pool Comparison")
  plot(input_file_names = ["res4", "res4, normalize data"], output_file_name = "Normalize Data Comparison")
  plot(input_file_names = ["res4", "res4, flip data", "res4, crop data", "res4, augment data"], output_file_name = "Augment Data Comparison")
  plot(input_file_names = ["res4", "res4, momentum optimizer"], output_file_name = "Momentum vs Adam Optimizer")