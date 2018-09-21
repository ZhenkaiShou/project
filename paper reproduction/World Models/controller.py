import numpy as np

import config

# Local hyperparameters
Z_LENGTH = config.Z_LENGTH
A_LENGTH = config.A_LENGTH
H_LENGTH = config.HIDDEN_UNITS
C_LENGTH = config.HIDDEN_UNITS
CONTROLLER_MODE = config.CONTROLLER_MODE

class Controller(object):
  def __init__(self):
    if CONTROLLER_MODE == "Z":
      self.weights = np.zeros((Z_LENGTH, A_LENGTH))
    elif CONTROLLER_MODE == "ZH":
      self.weights = np.zeros((Z_LENGTH + H_LENGTH, A_LENGTH))
    else:
      self.weights = np.zeros((Z_LENGTH + H_LENGTH + C_LENGTH, A_LENGTH))
    self.bias = np.zeros(A_LENGTH)
  
  def get_action(self, input):
    # range(a[0]) = [-1, 1]
    # range(a[1]) = [0, 1]
    # range(a[2]) = [0, 1]
    action = np.tanh(np.dot(input, self.weights) + self.bias)
    if np.shape(action) == 1:
      # If input is one dimension.
      action[1] = (action[1] + 1) / 2
      action[2] = np.clip(action[2], 0.0, 1.0)
    else:
      # If input is two dimension.
      action[:, 1] = (action[:, 1] + 1) / 2
      action[:, 2] = np.clip(action[:, 2], 0.0, 1.0)
    
    return action