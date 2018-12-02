import gym
import numpy as np
from collections import deque
from copy import copy
from gym import spaces
from PIL import Image

# Local hyperparameters
FRAME_SKIP = 4
FRAME_STACK = 4

def make_atari(env_name):
  env = gym.make(env_name)
  env = NoopResetEnv(env)
  env = MaxAndSkipEnv(env, skip = FRAME_SKIP)
  env = ProcessFrame84(env)
  env = FrameStack(env, k = FRAME_STACK)
  return env

'''
All the contents below are copied from 
  either baselines.common.atari_wrappers
  or https://github.com/openai/large-scale-curiosity/blob/master/wrappers.py
'''

def unwrap(env):
  if hasattr(env, "unwrapped"):
    return env.unwrapped
  elif hasattr(env, "env"):
    return unwrap(env.env)
  elif hasattr(env, "leg_env"):
    return unwrap(env.leg_env)
  else:
    return env

class NoopResetEnv(gym.Wrapper):
  def __init__(self, env, noop_max=30):
    """
    Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.
    """
    gym.Wrapper.__init__(self, env)
    self.noop_max = noop_max
    self.override_num_noops = None
    self.noop_action = 0
    assert env.unwrapped.get_action_meanings()[0] == 'NOOP'
  
  def reset(self, **kwargs):
    """ Do no-op action for a number of steps in [1, noop_max]."""
    self.env.reset(**kwargs)
    if self.override_num_noops is not None:
      noops = self.override_num_noops
    else:
      noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
    assert noops > 0
    obs = None
    for _ in range(noops):
      obs, _, done, _ = self.env.step(self.noop_action)
      if done:
        obs = self.env.reset(**kwargs)
    return obs
  
  def step(self, ac):
    return self.env.step(ac)

class MaxAndSkipEnv(gym.Wrapper):
  def __init__(self, env, skip=4):
    """Return only every `skip`-th frame"""
    gym.Wrapper.__init__(self, env)
    # most recent raw observations (for max pooling across time steps)
    self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
    self._skip       = skip
  
  def reset(self):
    return self.env.reset()
  
  def step(self, action):
    """Repeat action, sum reward, and max over last observations."""
    total_reward = 0.0
    done = None
    for i in range(self._skip):
      obs, reward, done, info = self.env.step(action)
      if i == self._skip - 2: self._obs_buffer[0] = obs
      if i == self._skip - 1: self._obs_buffer[1] = obs
      total_reward += reward
      if done:
        break
    # Note that the observation on the done=True frame doesn't matter
    max_frame = self._obs_buffer.max(axis=0)
    return max_frame, total_reward, done, info
  
  def reset(self, **kwargs):
    return self.env.reset(**kwargs)

class ProcessFrame84(gym.ObservationWrapper):
  def __init__(self, env, crop=True):
    self.crop = crop
    super(ProcessFrame84, self).__init__(env)
    self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
  
  def observation(self, obs):
    return ProcessFrame84.process(obs, crop=self.crop)
  
  @staticmethod
  def process(frame, crop=True):
    if frame.size == 210 * 160 * 3:
      img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
    elif frame.size == 250 * 160 * 3:
      img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
    elif frame.size == 224 * 240 * 3:  # mario resolution
      img = np.reshape(frame, [224, 240, 3]).astype(np.float32)
    else:
      assert False, "Unknown resolution." + str(frame.size)
    img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    size = (84, 110 if crop else 84)
    resized_screen = np.array(Image.fromarray(img).resize(size, resample=Image.BILINEAR), dtype=np.uint8)
    x_t = resized_screen[18:102, :] if crop else resized_screen
    x_t = np.reshape(x_t, [84, 84, 1])
    return x_t.astype(np.uint8)

class FrameStack(gym.Wrapper):
  def __init__(self, env, k):
    """
    Stack k last frames.
    Returns lazy array, which is much more memory efficient.
    See Also
    --------
    baselines.common.atari_wrappers.LazyFrames
    """
    gym.Wrapper.__init__(self, env)
    self.k = k
    self.frames = deque([], maxlen=k)
    shp = env.observation_space.shape
    self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8)
  
  def reset(self):
    ob = self.env.reset()
    for _ in range(self.k):
      self.frames.append(ob)
    return self._get_ob()
  
  def step(self, action):
    ob, reward, done, info = self.env.step(action)
    self.frames.append(ob)
    return self._get_ob(), reward, done, info
  
  def _get_ob(self):
    assert len(self.frames) == self.k
    return np.array(LazyFrames(list(self.frames)))

class LazyFrames(object):
  def __init__(self, frames):
    """
    This object ensures that common frames between the observations are only stored once.
    It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
    buffers.
    This object should only be converted to numpy array before being passed to the model.
    You'd not believe how complex the previous solution was.
    """
    self._frames = frames
    self._out = None
  
  def _force(self):
    if self._out is None:
      self._out = np.concatenate(self._frames, axis=2)
      self._frames = None
    return self._out
  
  def __array__(self, dtype=None):
    out = self._force()
    if dtype is not None:
      out = out.astype(dtype)
    return out
  
  def __len__(self):
    return len(self._force())
  
  def __getitem__(self, i):
    return self._force()[i]