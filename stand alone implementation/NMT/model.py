import tensorflow as tf

from config import *

class Seq2Seq_Model(object):
  def __init__(self):
    pass
  
  def create_single_cell(self, hidden_unit, dropout):
    cell = tf.nn.rnn_cell.LSTMCell(hidden_unit)
    if dropout > 0.0:
      cell = tf.contrib.rnn.DropoutWrapper(cell, 1.0 - dropout)
    return cell
  
  def build_training_model(self, source_vocab_length, target_vocab_length, source_input, target_input, target_output, source_length, target_length):
    # Create embedding.
    with tf.variable_scope("embedding"):
      embedding_encoder = tf.get_variable("embedding_encoder", [source_vocab_length, EMBEDDING_UNIT], dtype = tf.float32)
      embedding_decoder = tf.get_variable("embedding_decoder", [target_vocab_length, EMBEDDING_UNIT], dtype = tf.float32)
    
    with tf.variable_scope("Seq2Seq"):
      batch_size = tf.shape(source_input)[0]
      
      # Multi-layer Bidirectional LSTM as encoder.
      source_input_emb = tf.nn.embedding_lookup(embedding_encoder, source_input)
      encoder_cells_fw = [self.create_single_cell(HIDDEN_UNIT, DROPOUT) for _ in range(RNN_LAYER)]
      encoder_cells_bw = [self.create_single_cell(HIDDEN_UNIT, DROPOUT) for _ in range(RNN_LAYER)]
      encoder_cell_fw = tf.nn.rnn_cell.MultiRNNCell(encoder_cells_fw)
      encoder_cell_bw = tf.nn.rnn_cell.MultiRNNCell(encoder_cells_bw)
      encoder_initial_state_fw = encoder_cell_fw.zero_state(batch_size, tf.float32)
      encoder_initial_state_bw = encoder_cell_bw.zero_state(batch_size, tf.float32)
      # shape(encoder_output_fw) = (batch_size, time_length, hidden_unit)
      # encoder_state_fw = tuple(encoder_state_fw_layer_0, ..., encoder_state_fw_layer_n-1)
      #   encoder_state_fw_layer_i = LSTMStateTuple(encoder_state_fw_layer_i.c, encoder_state_fw_layer_i.h)
      #     shape(encoder_state_fw_layer_i.c) = (batch_size, hidden_unit)
      (encoder_output_fw, encoder_output_bw), (encoder_state_fw, encoder_state_bw) = tf.nn.bidirectional_dynamic_rnn(
        encoder_cell_fw, encoder_cell_bw, source_input_emb, source_length, encoder_initial_state_fw, encoder_initial_state_bw)
      
      # Concatenate the forward path and backward path for the encoder output and state.
      # shape(encoder_output) = (batch_size, time_length, 2 * hidden_unit)
      encoder_output = tf.concat([encoder_output_fw, encoder_output_bw], 2)
      # encoder_state = tuple(state_layer_0, ..., state_layer_n-1)
      #   state_layer_i = LSTMStateTuple(state_layer_i.c, state_layer_i.h)
      #     shape(state_layer_i.c) = (batch_size, 2 * hidden_unit)
      encoder_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(
        c = tf.concat([encoder_state_fw[i].c, encoder_state_bw[i].c], 1), 
        h = tf.concat([encoder_state_fw[i].h, encoder_state_bw[i].h], 1)) for i in range(RNN_LAYER)])
      
      # Multi-layer LSTM as decoder.
      target_input_emb = tf.nn.embedding_lookup(embedding_decoder, target_input)
      decoder_cells = [self.create_single_cell(2 * HIDDEN_UNIT, DROPOUT) for _ in range(RNN_LAYER)]
      decoder_cell = tf.nn.rnn_cell.MultiRNNCell(decoder_cells)
      
      # Apply attention mechanism on top of the RNN cell.
      '''
      In this implementation only Bahdanau Attention mechanism works because the number of hidden units in the decoder is doubled:
        number of hidden units in encoder = HIDDEN_UNIT
        number of hidden units in decoder = 2 * HIDDEN_UNIT
      This happens becasuse we concatenate the forward and backward output (as well as state) of the bidirectional LSTM encoder.
      However, in Luong Attention, the number of hidden units in both encoder and decoder should be the same.
      '''
      attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(HIDDEN_UNIT, encoder_output, source_length)
      decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, HIDDEN_UNIT, initial_cell_state = encoder_state)
      
      helper = tf.contrib.seq2seq.TrainingHelper(target_input_emb, target_length)
      decoder_initial_state = decoder_cell.zero_state(batch_size, tf.float32)
      output_layer = tf.layers.Dense(target_vocab_length, use_bias = False)
      decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, decoder_initial_state, output_layer)
      # decoder_output is the logits probability of each word in the target vocabulary.
      #   shape(decoder_output) = (batch_size, time_length, target_vocab_length)
      # sample_id is the argmax of decoder_output.
      #   shape(sample_id) = (batch_size, time_length)
      # decoder_state = tuple(state_layer_0, ..., state_layer_n)
      #   state_layer_i = LSTMStateTuple(state_layer_i.c, state_layer_i.h)
      #     shape(state_layer_i.c) = (batch_size, 2 * hidden_unit)
      (decoder_output, sample_id), decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
    
    self.embedding = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "embedding")
    self.network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "Seq2Seq")
    
    # Loss.
    mask_function = lambda length: tf.concat([tf.tile([1.0], [length]), tf.tile([0.0], [tf.reduce_max(target_length) - length])], 0)
    weights_mask = tf.map_fn(mask_function, target_length, dtype = tf.float32)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = target_output, logits = decoder_output)
    self.loss = tf.reduce_mean(tf.reduce_sum(cross_entropy * weights_mask, 1))
    
    # Additional loss for measuring training progress (not used for training).
    self.avg_loss = tf.reduce_sum(cross_entropy * weights_mask) / tf.reduce_sum(tf.cast(target_length, tf.float32))
    
    # Optimization.
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE) 
    self.train_op = optimizer.minimize(self.loss)
  
  def build_evaluation_model(self, source_vocab_length, target_vocab_length, target_sos_id, target_eos_id, source_input, target_output, source_length):
    # Create embedding.
    with tf.variable_scope("embedding"):
      embedding_encoder = tf.get_variable("embedding_encoder", [source_vocab_length, EMBEDDING_UNIT], dtype = tf.float32)
      embedding_decoder = tf.get_variable("embedding_decoder", [target_vocab_length, EMBEDDING_UNIT], dtype = tf.float32)
    
    with tf.variable_scope("Seq2Seq"):
      batch_size = tf.shape(source_input)[0]
      
      # Multi-layer Bidirectional LSTM as encoder.
      source_input_emb = tf.nn.embedding_lookup(embedding_encoder, source_input)
      encoder_cells_fw = [self.create_single_cell(HIDDEN_UNIT, 0.0) for _ in range(RNN_LAYER)]
      encoder_cells_bw = [self.create_single_cell(HIDDEN_UNIT, 0.0) for _ in range(RNN_LAYER)]
      encoder_cell_fw = tf.nn.rnn_cell.MultiRNNCell(encoder_cells_fw)
      encoder_cell_bw = tf.nn.rnn_cell.MultiRNNCell(encoder_cells_bw)
      encoder_initial_state_fw = encoder_cell_fw.zero_state(batch_size, tf.float32)
      encoder_initial_state_bw = encoder_cell_bw.zero_state(batch_size, tf.float32)
      # shape(encoder_output_fw) = (batch_size, time_length, hidden_unit)
      # encoder_state_fw = tuple(encoder_state_fw_layer_0, ..., encoder_state_fw_layer_n-1)
      #   encoder_state_fw_layer_i = LSTMStateTuple(encoder_state_fw_layer_i.c, encoder_state_fw_layer_i.h)
      #     shape(encoder_state_fw_layer_i.c) = (batch_size, hidden_unit)
      (encoder_output_fw, encoder_output_bw), (encoder_state_fw, encoder_state_bw) = tf.nn.bidirectional_dynamic_rnn(
        encoder_cell_fw, encoder_cell_bw, source_input_emb, source_length, encoder_initial_state_fw, encoder_initial_state_bw)
      
      # Concatenate the forward path and backward path for the encoder output and state.
      # shape(encoder_output) = (batch_size, time_length, 2 * hidden_unit)
      encoder_output = tf.concat([encoder_output_fw, encoder_output_bw], 2)
      # encoder_state = tuple(state_layer_0, ..., state_layer_n-1)
      #   state_layer_i = LSTMStateTuple(state_layer_i.c, state_layer_i.h)
      #     shape(state_layer_i.c) = (batch_size, 2 * hidden_unit)
      encoder_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(
        c = tf.concat([encoder_state_fw[i].c, encoder_state_bw[i].c], 1), 
        h = tf.concat([encoder_state_fw[i].h, encoder_state_bw[i].h], 1)) for i in range(RNN_LAYER)])
      
      # Multi-layer LSTM as decoder.
      decoder_cells = [self.create_single_cell(2 * HIDDEN_UNIT, 0.0) for _ in range(RNN_LAYER)]
      decoder_cell = tf.nn.rnn_cell.MultiRNNCell(decoder_cells)
      
      # Tile the tensors for beam search.
      tiled_encoder_output = tf.contrib.seq2seq.tile_batch(encoder_output, BEAM_WIDTH)
      tiled_encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, BEAM_WIDTH)
      tiled_source_length = tf.contrib.seq2seq.tile_batch(source_length, BEAM_WIDTH)
      
      # Apply attention mechanism on top of the RNN cell.
      attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(HIDDEN_UNIT, tiled_encoder_output, tiled_source_length)
      decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, HIDDEN_UNIT, initial_cell_state = tiled_encoder_state)
      
      decoder_initial_state = decoder_cell.zero_state(batch_size * BEAM_WIDTH, tf.float32)
      output_layer = tf.layers.Dense(target_vocab_length, use_bias = False)
      decoder = tf.contrib.seq2seq.BeamSearchDecoder(
        cell = decoder_cell, embedding = embedding_decoder, 
        start_tokens = tf.tile([target_sos_id], [batch_size]), end_token = target_eos_id, 
        initial_state = decoder_initial_state, 
        beam_width = BEAM_WIDTH, output_layer = output_layer, 
        coverage_penalty_weight = COVERAGE_PENALTY)
      # predicted_ids is the final output of beam search, which is ordered from the best to the worst.
      #   shape(predicted_ids) = (batch_size, time_length, beam_width)
      (predicted_ids, _), _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations = MAX_SEQUENCE_LENGTH)
      
      # Output IDs for evaluation.
      self.translation_ids = predicted_ids[:, :, 0]
      self.reference_ids = target_output
    
    self.embedding = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "embedding")
    self.network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "Seq2Seq")

  def build_test_model(self, source_vocab_length, target_vocab_length, source_string_table, target_string_table, target_sos_id, target_eos_id, source_input, source_length):
    # Create embedding.
    with tf.variable_scope("embedding"):
      embedding_encoder = tf.get_variable("embedding_encoder", [source_vocab_length, EMBEDDING_UNIT], dtype = tf.float32)
      embedding_decoder = tf.get_variable("embedding_decoder", [target_vocab_length, EMBEDDING_UNIT], dtype = tf.float32)
    
    with tf.variable_scope("Seq2Seq"):
      batch_size = tf.shape(source_input)[0]
      
      # Multi-layer Bidirectional LSTM as encoder.
      source_input_emb = tf.nn.embedding_lookup(embedding_encoder, source_input)
      encoder_cells_fw = [self.create_single_cell(HIDDEN_UNIT, 0.0) for _ in range(RNN_LAYER)]
      encoder_cells_bw = [self.create_single_cell(HIDDEN_UNIT, 0.0) for _ in range(RNN_LAYER)]
      encoder_cell_fw = tf.nn.rnn_cell.MultiRNNCell(encoder_cells_fw)
      encoder_cell_bw = tf.nn.rnn_cell.MultiRNNCell(encoder_cells_bw)
      encoder_initial_state_fw = encoder_cell_fw.zero_state(batch_size, tf.float32)
      encoder_initial_state_bw = encoder_cell_bw.zero_state(batch_size, tf.float32)
      # shape(encoder_output_fw) = (batch_size, time_length, hidden_unit)
      # encoder_state_fw = tuple(encoder_state_fw_layer_0, ..., encoder_state_fw_layer_n-1)
      #   encoder_state_fw_layer_i = LSTMStateTuple(encoder_state_fw_layer_i.c, encoder_state_fw_layer_i.h)
      #     shape(encoder_state_fw_layer_i.c) = (batch_size, hidden_unit)
      (encoder_output_fw, encoder_output_bw), (encoder_state_fw, encoder_state_bw) = tf.nn.bidirectional_dynamic_rnn(
        encoder_cell_fw, encoder_cell_bw, source_input_emb, source_length, encoder_initial_state_fw, encoder_initial_state_bw)
      
      # Concatenate the forward path and backward path for the encoder output and state.
      # shape(encoder_output) = (batch_size, time_length, 2 * hidden_unit)
      encoder_output = tf.concat([encoder_output_fw, encoder_output_bw], 2)
      # encoder_state = tuple(state_layer_0, ..., state_layer_n-1)
      #   state_layer_i = LSTMStateTuple(state_layer_i.c, state_layer_i.h)
      #     shape(state_layer_i.c) = (batch_size, 2 * hidden_unit)
      encoder_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(
        c = tf.concat([encoder_state_fw[i].c, encoder_state_bw[i].c], 1), 
        h = tf.concat([encoder_state_fw[i].h, encoder_state_bw[i].h], 1)) for i in range(RNN_LAYER)])
      
      # Multi-layer LSTM as decoder.
      decoder_cells = [self.create_single_cell(2 * HIDDEN_UNIT, 0.0) for _ in range(RNN_LAYER)]
      decoder_cell = tf.nn.rnn_cell.MultiRNNCell(decoder_cells)
      
      # Tile the tensors for beam search.
      tiled_encoder_output = tf.contrib.seq2seq.tile_batch(encoder_output, BEAM_WIDTH)
      tiled_encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, BEAM_WIDTH)
      tiled_source_length = tf.contrib.seq2seq.tile_batch(source_length, BEAM_WIDTH)
      
      # Apply attention mechanism on top of the RNN cell.
      attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(HIDDEN_UNIT, tiled_encoder_output, tiled_source_length)
      decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, HIDDEN_UNIT, initial_cell_state = tiled_encoder_state)
      
      decoder_initial_state = decoder_cell.zero_state(batch_size * BEAM_WIDTH, tf.float32)
      output_layer = tf.layers.Dense(target_vocab_length, use_bias = False)
      decoder = tf.contrib.seq2seq.BeamSearchDecoder(
        cell = decoder_cell, embedding = embedding_decoder, 
        start_tokens = tf.tile([target_sos_id], [batch_size]), end_token = target_eos_id, 
        initial_state = decoder_initial_state, 
        beam_width = BEAM_WIDTH, output_layer = output_layer, 
        coverage_penalty_weight = COVERAGE_PENALTY)
      # predicted_ids is the final output of beam search, which is ordered from the best to the worst
      #   shape(predicted_ids) = (batch_size, time_length, beam_width)
      (predicted_ids, _), _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations = MAX_SEQUENCE_LENGTH)
      
      # Output sentence.
      self.source_sentence = source_string_table.lookup(tf.cast(source_input, tf.int64))
      self.translation_sentence = target_string_table.lookup(tf.cast(predicted_ids[:, :, 0], tf.int64))
    
    self.embedding = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "embedding")
    self.network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = "Seq2Seq")