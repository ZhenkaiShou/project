import collections
import tensorflow as tf

from config import *

def load_file(file):
  data = []
  with open(DATA_DIR + file, "r", encoding = "utf-8") as f:
    for word in f:
      data.append(word.strip())
  length = len(data)
  
  return data, length

def load_and_check_vocab(vocab_file):
  # Check whether UNK, SOS, EOS appears in a vocabulary.
  vocab, length = load_file(vocab_file)
  if UNK not in vocab or SOS not in vocab or EOS not in vocab:
    # If not, add them to the beginning of the vocabulary.
    with open(DATA_DIR + vocab_file, "w", encoding = "utf-8") as f:
      vocab = [UNK, SOS, EOS] + vocab
      for word in vocab:
        f.write(word + "\n")
      length += 3
  
  return vocab, length

def check_dataset(data_file):
  # Return True if the length of a sentence is between (1, MAX_SEQUENCE_LENGTH].
  data, _ = load_file(data_file)
  bool_mask = [len(sentence.split()) > 0 and len(sentence.split()) <= MAX_SEQUENCE_LENGTH for sentence in data]
  
  return bool_mask

def get_dataset_iterator(dataset_length, source_data_file, target_data_file, source_vocab_table, target_vocab_table, source_eos_id, target_sos_id, target_eos_id):
  # Create dataset.
  source_dataset = tf.data.TextLineDataset(DATA_DIR + source_data_file)
  target_dataset = tf.data.TextLineDataset(DATA_DIR + target_data_file)
  dataset = tf.data.Dataset.zip((source_dataset, target_dataset))
  
  # Shuffle and repeat dataset.
  dataset = dataset.shuffle(dataset_length, reshuffle_each_iteration = True)
  dataset = dataset.repeat(EPOCH)
  
  # Split a sentence into a list of strings.
  dataset = dataset.map(lambda source, target: (tf.string_split([source]).values, tf.string_split([target]).values))
  
  # Filter data whose length is not in range of (0, MAX_SEQUENCE_LENGTH].
  dataset = dataset.filter(lambda source, target: tf.logical_and(
    tf.logical_and(tf.size(source) > 0, tf.size(source) <= MAX_SEQUENCE_LENGTH), 
    tf.logical_and(tf.size(target) > 0, tf.size(target) <= MAX_SEQUENCE_LENGTH)))
  
  # Convert string into index.
  dataset = dataset.map(lambda source, target: (tf.cast(source_vocab_table.lookup(source), tf.int32), tf.cast(target_vocab_table.lookup(target), tf.int32)))
  
  # Create a target input prefixed with SOS and a target output postfixed with EOS.
  dataset = dataset.map(lambda source, target: (source, tf.concat([[target_sos_id], target], 0), tf.concat([target, [target_eos_id]], 0)))
  
  # Add sequence length.
  dataset = dataset.map(lambda source, target_in, target_out: (source, target_in, target_out, tf.size(source), tf.size(target_in)))
  dataset = dataset.prefetch(1)
  
  # Pad dataset to the maximum length in each bucket batch.
  bucket = tf.contrib.data.bucket_by_sequence_length(
    element_length_func = lambda source_data, target_input_data, target_output_data, source_length, target_length: tf.maximum(source_length, target_length),
    bucket_boundaries = [int(MAX_SEQUENCE_LENGTH * (BUCKET_DECAY ** (BUCKET_NUM - i))) for i in range(BUCKET_NUM)],
    bucket_batch_sizes = [BUCKET_BATCH_SIZE] * (BUCKET_NUM + 1),
    padded_shapes = (tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([]), tf.TensorShape([])),
    padding_values = (source_eos_id, target_eos_id, target_eos_id, 0, 0))
  dataset = dataset.apply(bucket)
  
  # Define iterator.
  iterator = dataset.make_initializable_iterator()
  initializer = iterator.initializer
  source_input, target_input, target_output, source_length, target_length = iterator.get_next()
  
  output_tuple = collections.namedtuple("output_tuple", ("initializer", "source_input", "target_input", "target_output", "source_length", "target_length"))
  output = output_tuple(initializer = initializer, source_input = source_input, target_input = target_input, target_output = target_output, source_length = source_length, target_length = target_length)
  
  return output

def get_test_dataset_iterator(source_data_file, source_vocab_file, source_eos_id):
  # Compute the length of dataset.
  _, dataset_length = load_file(source_data_file)
  
  # Create dataset.
  dataset = tf.data.TextLineDataset(DATA_DIR + source_data_file)
  
  # Shuffle dataset.
  dataset = dataset.shuffle(dataset_length, reshuffle_each_iteration = True)
  
  # Split a sentence into a list of strings.
  dataset = dataset.map(lambda source: tf.string_split([source]).values)
  
  # Convert string into index.
  dataset = dataset.map(lambda source: tf.cast(source_vocab_table.lookup(source), tf.int32))
  
  # Add sequence length.
  dataset = dataset.map(lambda source: (source, tf.size(source)))
  dataset = dataset.prefetch(1)
  
  # Pad dataset to the maximum length in each bucket batch.
  bucket = tf.contrib.data.bucket_by_sequence_length(
    element_length_func = lambda source_data, source_length: source_length,
    bucket_boundaries = BUCKET_BOUNDARIES,
    bucket_batch_sizes = [BUCKET_BATCH_SIZE] * (len(BUCKET_BOUNDARIES) + 1),
    padded_shapes = (tf.TensorShape([None]), tf.TensorShape([])),
    padding_values = (source_eos_id, 0))
  dataset = dataset.apply(bucket)
  
  # Define iterator.
  iterator = dataset.make_initializable_iterator()
  initializer = iterator.initializer
  source_input, source_length = iterator.get_next()
  
  output_tuple = collections.namedtuple("output_tuple", ("initializer", "source_input", "source_length"))
  output = output_tuple(initializer = initializer, source_input = source_input, source_length = source_length)
  
  return output