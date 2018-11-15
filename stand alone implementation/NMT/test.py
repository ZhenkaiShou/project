import numpy as np
import os
import tensorflow as tf

from config import *
from model import Seq2Seq_Model
from util import load_and_check_vocab, check_input, get_input_data, decode_byte_sequence, shorten_sequence, concatenate_strings

def test(input_sentence):
  # Load and check whether UNK, SOS, EOS appears in the vocabulary.
  source_vocab, source_vocab_length = load_and_check_vocab(SOURCE_VOCAB_FILE)
  target_vocab, target_vocab_length = load_and_check_vocab(TARGET_VOCAB_FILE)
  
  # Get UNK, SOS, EOS id.
  source_unk_id = source_vocab.index(UNK)
  target_sos_id = target_vocab.index(SOS)
  target_eos_id = target_vocab.index(EOS)
  
  # Filter out the effective input sentence.
  bool_mask = check_input(input_sentence)
  input_sentence = [sentence for sentence, bool_value in zip(input_sentence, bool_mask) if bool_value]
  
  # Create source vocabulary table.
  source_vocab_table = tf.contrib.lookup.index_table_from_file(DATA_DIR + SOURCE_VOCAB_FILE, default_value = source_unk_id)
  
  # Create source and target string tables.
  source_string_table = tf.contrib.lookup.index_to_string_table_from_file(DATA_DIR + SOURCE_VOCAB_FILE, default_value = UNK)
  target_string_table = tf.contrib.lookup.index_to_string_table_from_file(DATA_DIR + TARGET_VOCAB_FILE, default_value = UNK)
  
  # Get input data tuple(source_input, source_length).
  Input_Sentence = tf.placeholder(tf.string, (None,))
  input_data = get_input_data(Input_Sentence, source_vocab_table)
  
  # Load model.
  model = Seq2Seq_Model()
  model.build_test_model(source_vocab_length, target_vocab_length, source_string_table, target_string_table, target_sos_id, target_eos_id, input_data.source_input, input_data.source_length)
  
  table_initializer = tf.tables_initializer()
  
  with tf.Session() as sess:
    # Initialization.
    sess.run(table_initializer)
    
    # Load parameters.
    saver_embedding = tf.train.Saver(model.embedding)
    saver_network = tf.train.Saver(model.network_params)
    saver_embedding.restore(sess, SAVE_DIR + "embedding")
    saver_network.restore(sess, SAVE_DIR + "network_params")
    
    for i in range(len(input_sentence)):
      # Feed the input sentence to the network one by one.
      source_sentence, translation_sentence = sess.run([model.source_sentence, model.translation_sentence], feed_dict = {Input_Sentence: [input_sentence[i]]})
      
      # Decode the byte array to string list.
      source_sentence = decode_byte_sequence(source_sentence)
      translation_sentence = decode_byte_sequence(translation_sentence)
      
      # Shorten the sentence.
      source_sentence = shorten_sequence(source_sentence, EOS, dtype = str)
      translation_sentence = shorten_sequence(translation_sentence, EOS, dtype = str)
      
      # Concatenate a list of strings into a single string for better visualization.
      source_sentence = concatenate_strings(source_sentence)
      translation_sentence = concatenate_strings(translation_sentence)
      
      source_sentence = source_sentence[0]
      translation_sentence = translation_sentence[0]
      
      if len(input_sentence) > 1:
        print("Sentence ", i + 1, ": ", sep = "")
        print("  Source: ", source_sentence, sep = "")
        print("  Translation: ", translation_sentence, sep = "")
      else:
        print("Source: ", source_sentence, sep = "")
        print("Translation: ", translation_sentence, sep = "")

if __name__ == "__main__":
  input_sentence = ["This is a test list .", "It may contain more than one sentence .", "Any unknown words will be transformed into blablabla ."]
  test(input_sentence)