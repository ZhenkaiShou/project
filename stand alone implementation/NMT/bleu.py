import collections
import numpy as np

def get_ngrams(sequence, max_order):
  sequence_length = len(sequence)
  ngrams = collections.Counter()
  for order in range(max_order):
    for i in range(sequence_length - order):
      element = tuple(sequence[i: i+order+1])
      ngrams[element] += 1
  
  return ngrams

def get_bleu_score(translation, references, max_order = 4):
  # Compute the BLEU score for the translation given the corresponding list of references.
  # Get the translation and reference length.
  trans_length = len(translation)
  refer_length = np.amin([len(reference) for reference in references])
  
  # Get the n-gram upto max_order for both translation and reference.
  trans_ngrams = get_ngrams(translation, max_order)
  refer_ngrams = collections.Counter()
  for reference in references:
    refer_ngrams |= get_ngrams(reference, max_order)
  
  # Get the clipped translation n-gram.
  clipped_trans_ngrams = trans_ngrams & refer_ngrams
  
  # Compute the precision.
  trans_counts = [0 for _ in range(max_order)]
  for element in trans_ngrams:
    trans_counts[len(element) - 1] += trans_ngrams[element]
  clipped_trans_counts = [0 for _ in range(max_order)]
  for element in clipped_trans_ngrams:
    clipped_trans_counts[len(element) - 1] += clipped_trans_ngrams[element]
  precision = [clipped_trans_count / trans_count if trans_count > 0 else -1 for clipped_trans_count, trans_count in zip(clipped_trans_counts, trans_counts)]
  
  # Average the geometric mean. Here we ignore the case where precision = -1, 0.
  log_sum = 0
  for value in precision:
    if value > 0:
      log_sum += 1.0 / max_order * np.log(value)
  geo_mean = np.exp(log_sum)
  
  # Compute the brevity penalty.
  brevity_penalty = np.minimum(np.exp(1.0 - refer_length / trans_length), 1.0)
  
  # Compute the bleu score.
  bleu_score = brevity_penalty * geo_mean
  
  return bleu_score