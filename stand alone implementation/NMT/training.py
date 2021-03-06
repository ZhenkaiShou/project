import numpy as np
import os
import tensorflow as tf

from buffer import Buffer
from config import *
from download import download
from model import Seq2Seq_Model
from util import load_and_check_vocab, check_dataset, get_dataset_iterator

def train():
  # Download dataset.
  download()
  
  # Create folders.
  if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR)
  
  # Load and check whether UNK, SOS, EOS appears in the vocabulary.
  source_vocab, source_vocab_length = load_and_check_vocab(SOURCE_VOCAB_FILE)
  target_vocab, target_vocab_length = load_and_check_vocab(TARGET_VOCAB_FILE)
  
  # Get UNK, SOS, EOS id.
  source_unk_id = source_vocab.index(UNK)
  source_eos_id = source_vocab.index(EOS)
  target_unk_id = target_vocab.index(UNK)
  target_sos_id = target_vocab.index(SOS)
  target_eos_id = target_vocab.index(EOS)
  
  # Compute the length of effective training data.
  source_bool_mask = check_dataset(SOURCE_TRAINING_FILE)
  target_bool_mask = check_dataset(TARGET_TRAINING_FILE)
  bool_mask = [x and y for x, y in zip(source_bool_mask, target_bool_mask)]
  dataset_length = sum(bool_mask)
  
  # Create source and target vocabulary tables.
  source_vocab_table = tf.contrib.lookup.index_table_from_file(DATA_DIR + SOURCE_VOCAB_FILE, default_value = source_unk_id)
  target_vocab_table = tf.contrib.lookup.index_table_from_file(DATA_DIR + TARGET_VOCAB_FILE, default_value = target_unk_id)
  
  # Get dataset iterator tuple(initializer, source_input, target_input, target_output, source_length, target_length).
  iterator = get_dataset_iterator(dataset_length, SOURCE_TRAINING_FILE, TARGET_TRAINING_FILE, source_vocab_table, target_vocab_table, source_eos_id, target_sos_id, target_eos_id)
  
  # Load model.
  model = Seq2Seq_Model()
  model.build_training_model(source_vocab_length, target_vocab_length, iterator.source_input, iterator.target_input, iterator.target_output, iterator.source_length, iterator.target_length)
  
  iterator_initializer = iterator.initializer
  table_initializer = tf.tables_initializer()
  variable_initializer = tf.global_variables_initializer()
  
  with tf.Session() as sess:
    # Initialization.
    sess.run([iterator_initializer, table_initializer, variable_initializer])
    
    avg_loss_buffer = Buffer(2000)
    for step in range(TRAINING_STEP):
      _, loss, avg_loss = sess.run([model.train_op, model.loss, model.avg_loss])
      avg_loss_buffer.append(avg_loss)
      buffer_size = avg_loss_buffer.get_size()
      avg_loss_ = np.mean(avg_loss_buffer.get_buffer())
      print("Current progress: ", step+1, "/", TRAINING_STEP, ". Average loss over the latest ", buffer_size, " training steps: ", format(avg_loss_, ".8f"), sep = "", end = "\r", flush = True)
    
    # Save parameters.
    saver_embedding = tf.train.Saver(model.embedding)
    saver_network = tf.train.Saver(model.network_params)
    saver_embedding.save(sess, SAVE_DIR + "embedding")
    saver_network.save(sess, SAVE_DIR + "network_params")
  
if __name__ == "__main__":
  train()