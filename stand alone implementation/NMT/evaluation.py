import numpy as np
import os
import tensorflow as tf

from bleu import get_bleu_score
from config import *
from model import Seq2Seq_Model
from util import load_and_check_vocab, check_dataset, get_dataset_iterator, shorten_sequence

def evaluate():
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
  iterator = get_dataset_iterator(dataset_length, SOURCE_TEST_FILE_1, TARGET_TEST_FILE_1, source_vocab_table, target_vocab_table, source_eos_id, target_sos_id, target_eos_id)
  
  # Load model.
  model = Seq2Seq_Model()
  model.build_evaluation_model(source_vocab_length, target_vocab_length, target_sos_id, target_eos_id, iterator.source_input, iterator.target_output, iterator.source_length)
  
  iterator_initializer = iterator.initializer
  table_initializer = tf.tables_initializer()
  
  with tf.Session() as sess:
    # Initialization.
    sess.run([iterator_initializer, table_initializer])
    
    # Load parameters.
    saver_embedding = tf.train.Saver(model.embedding)
    saver_network = tf.train.Saver(model.network_params)
    saver_embedding.restore(sess, SAVE_DIR + "embedding")
    saver_network.restore(sess, SAVE_DIR + "network_params")
    
    list_translation = []
    list_reference = []
    
    total_step = int(np.ceil(dataset_length / BUCKET_BATCH_SIZE))
    for step in range(total_step):
      translation_ids, reference_ids = sess.run([model.translation_ids, model.reference_ids])
      
      # Shorten the sequence by removing the paddings.
      translation_ids = shorten_sequence(translation_ids, target_eos_id, dtype = np.int32)
      reference_ids = shorten_sequence(reference_ids, target_eos_id, dtype = np.int32)
      
      list_translation += translation_ids
      list_reference += [[reference] for reference in reference_ids]
      print("Current progress: ", step+1, "/", total_step, sep = "", end = "\r", flush = True)
    
    # Estimate the BLEU score.
    score = get_bleu_score(list_translation, list_reference)
    
    print("")
    print("BLEU score = ", format(score, ".8f"), sep = "")

if __name__ == "__main__":
  evaluate()