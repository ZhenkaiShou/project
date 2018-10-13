import numpy as np
import random

class ReplayBuffer(object):
  def __init__(self, buffer_size):
    self.buffer_size = buffer_size
    self.buffer_list = []
  
  def append(self, item):
    if len(self.buffer_list) >= self.buffer_size:
      self.buffer_list.pop(0)
    self.buffer_list.append(item)
  
  def sample(self, batch_size):
    return random.sample(self.buffer_list, batch_size)

  def get_size(self):
    return len(self.buffer_list)