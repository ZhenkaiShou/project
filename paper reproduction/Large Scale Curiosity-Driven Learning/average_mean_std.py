import numpy as np

def average_mean_std(mean_a, std_a, count_a, mean_b, std_b, count_b):
  # Compute the mean and standard deviation of two groups of distribution.
  new_count = count_a + count_b
  new_mean = (mean_a * count_a + mean_b * count_b) / new_count
  new_std = np.sqrt(((std_a**2 + (mean_a - new_mean)**2) * count_a + (std_b**2 + (mean_b - new_mean)**2) * count_b) / new_count)
  return new_mean, new_std, new_count