import numpy as np

class PrioritizedReplayBuffer(object):
  def __init__(self, buffer_size, alpha = 1.0, epsilon = 1e-6):
    self.buffer_size = buffer_size
    self.alpha = alpha
    self.epsilon = epsilon
    
    self.next_index = 0
    self.filled_size = 0
    self.tree_size = 2 * buffer_size - 1
    self.data = np.empty(buffer_size, dtype = object)
    self.priority_tree = np.zeros(self.tree_size)
    self.priority_max = 1.0
  
  def append(self, data):
    # Append new data into the replay buffer.
    self.data[self.next_index] = data
    self._update_priority(self.next_index, self.priority_max)
    self.next_index = (self.next_index + 1) % self.buffer_size
    self.filled_size = np.minimum(self.filled_size + 1, self.buffer_size)
  
  def _update_priority(self, data_index, priority):
    # Update the priority of a data index.
    tree_index = self._get_tree_index(data_index)
    delta_priority = np.power(priority, self.alpha) - self.priority_tree[tree_index]
    while tree_index >= 0:
      self.priority_tree[tree_index] += delta_priority
      tree_index = (tree_index - 1) // 2
    self.priority_max = np.maximum(self.priority_max, priority)
  
  def _get_tree_index(self, data_index):
    # Transform data index into tree index.
    return self.buffer_size - 1 + data_index
  
  def _get_data_index(self, tree_index):
    # Transform tree index into data index.
    return tree_index - self.buffer_size + 1
  
  def sample(self, batch_size, beta):
    # Sample a minibatch of data.
    priority_sum = self.priority_tree[0]
    priority_interval = priority_sum / batch_size
    random_value = priority_interval * (np.arange(0, batch_size) + np.random.uniform(size = batch_size))
    sampled_leaf_index = np.array([self._search_leaf_index(random_value[i]) for i in range(batch_size)])
    sampled_data_index = self._get_data_index(tree_index = sampled_leaf_index)
    sampled_data = self.data[sampled_data_index]
    sampled_priority = self.priority_tree[sampled_leaf_index]
    # Compute importance sampling weights.
    #   p_uniform = 1 / N
    #   p_prioritized = priority / priority_sum
    #   weights = (p_uniform / p_prioritized)^(beta)
    #   norm = max(weights) = max[(p_uniform / p_prioritized)^(beta)] = (p_uniform / min[p_prioritized])^(beta)
    #   normalized_weights = weights / norm 
    #     = (p_uniform / p_prioritized)^(beta) / (p_uniform / min[p_prioritized])^(beta)
    #     = (min[p_prioritized] / p_prioritized)^(beta)
    #     = (min[priority] / priority)^(beta)
    priority_min = np.amin(self.priority_tree[(self.tree_size - self.buffer_size) : (self.tree_size - self.buffer_size + self.filled_size)])
    sampled_weights = np.power(priority_min / sampled_priority, beta)
    
    return (sampled_data_index, sampled_data, sampled_weights)
  
  def _search_leaf_index(self, value):
    # Search the leaf node of a given value.
    tree_index = 0
    while tree_index < self.buffer_size - 1:
      left_child_index = 2 * (tree_index + 1) - 1
      right_child_index = left_child_index + 1
      if self.priority_tree[left_child_index] > value:
       tree_index = left_child_index
      else:
       value -= self.priority_tree[left_child_index]
       tree_index = right_child_index
    return tree_index
  
  def update_priorities(self, data_indices, values):
    # Update the priorities of a batch of data.
    priorities = np.abs(values) + self.epsilon
    for data_index, priority in zip(data_indices, priorities):
      self._update_priority(data_index, priority)