import numpy as np

class LinearSchedule(object):
  def __init__(self, start, end, decay_step):
    """
    The value changes linearly:
      value(step) = start + (end - start) * np.clip(step / decay_step, 0, 1)
    """
    self.start = start
    self.end = end
    self.decay_step = decay_step
  
  def get_value(self, step):
    return self.start + (self.end - self.start) * np.clip(step / self.decay_step, 0, 1)

class StaircaseSchedule(object):
  def __init__(self, values, cutoff_steps):
    """
    The value changes with cutoff:
      value(step) = values[0] if step < cutoff_steps[0]
                  = values[1] if cutoff_steps[0] <= step < cutoff_steps[1]
                  = ...
    """
    self.values = values
    self.cutoff_steps = cutoff_steps
  
  def get_value(self, step):
    return self.values[np.sum(step >= np.array(self.cutoff_steps))]