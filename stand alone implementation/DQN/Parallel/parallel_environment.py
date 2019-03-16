import multiprocessing as mp
import numpy as np

def run_subprocess(pipe, env):
  while True:
    # Wait for the next command.
    cmd, data = pipe.recv()
    if cmd == "reset":
      obs = env.reset()
      pipe.send(obs)
    if cmd == "step":
      obs, reward, done, info = env.step(data)
      if done:
        obs = env.reset()
      pipe.send((obs, reward, done, info))
    if cmd == "close":
      pipe.send(None)
      pipe.close()
      env.close()
      break

class ParallelEnvironment(object):
  def __init__(self, list_env):
    self.list_env = list_env
    self.list_pipe_parent = []
    self.last_response = None
    
    # Create a subprocess for each environment.
    for env in list_env:
      pipe_parent, pipe_child = mp.Pipe()
      process = mp.Process(target = run_subprocess, args = (pipe_child, env))
      process.start()
      pipe_child.close()
      self.list_pipe_parent.append(pipe_parent)
  
  def get_last_response(self):
    return self.last_response
  
  def reset(self):
    # Reset all environments.
    for pipe_parent in self.list_pipe_parent:
      content = ("reset", None)
      pipe_parent.send(content)
    # Wait for response.
    obs = [pipe_parent.recv() for pipe_parent in self.list_pipe_parent]
    self.last_response = obs
    return obs
  
  def step(self, list_action):
    # Take an action for each environment.
    for pipe_parent, action in zip(self.list_pipe_parent, list_action):
      content = ("step", action)
      pipe_parent.send(content)
    # Wait for response.
    response = [pipe_parent.recv() for pipe_parent in self.list_pipe_parent]
    obs, reward, done, info = map(list, zip(*response))
    self.last_response = (obs, reward, done, info)
    return obs, reward, done, info
  
  def close(self):
    # Close all environments.
    for pipe_parent in self.list_pipe_parent:
      content = ("close", None)
      pipe_parent.send(content)
    # Wait for response.
    [pipe_parent.recv() for pipe_parent in self.list_pipe_parent]
    self.last_response = None
    # Close all pipes.
    for pipe_parent in self.list_pipe_parent:
      pipe_parent.close()