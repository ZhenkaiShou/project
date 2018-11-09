import numpy as np
import os
import tensorflow as tf

from config import *
from download import download
from model import *
from util import load_file, load_and_check_vocab, check_dataset, get_dataset_iterator

def training():
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
  
  # Create soruce and target vocabulary tables.
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
    
    for epoch in range(EPOCH):
      list_avg_loss = []
      for step in range(int(np.ceil(dataset_length / BUCKET_BATCH_SIZE))):
        _, loss, avg_loss = sess.run([model.train_op, model.loss, model.avg_loss])
        list_avg_loss.append(avg_loss)
        
        if step % 10 == 0:
          print("Epoch ", format(epoch, "02d"), ", Step ", format(step, "04d"), ":", sep = "")
          print("  Loss = ", format(loss, ".8f"), ", Avg Loss = ", format(avg_loss, ".8f"), sep = "")
      print("Average Loss for Epoch ", format(epoch, "02d"), ": ", format(np.mean(list_avg_loss), ".8f"), sep = "")
    
    # Save parameters.
    saver_embedding = tf.train.Saver(model.embedding)
    saver_network = tf.train.Saver(model.network_params)
    saver_embedding.save(sess, SAVE_DIR + "embedding")
    saver_network.save(sess, SAVE_DIR + "network_params")
  
if __name__ == "__main__":
  training()