import math
import numpy as np
import os

from gym import utils
from mujoco_env import MyMujocoEnv
from gym.envs.mujoco.reacher import ReacherEnv

class MyReacherEnv(MyMujocoEnv, utils.EzPickle):
  def __init__(self):
    utils.EzPickle.__init__(self)
    MyMujocoEnv.__init__(self, "./reacher.xml", 2)
  
  def step(self, a):
    obs = self._get_obs()
    done = False
    self.do_simulation(a, self.frame_skip)
    
    return obs, None, done, None
    
  def viewer_setup(self):
    self.viewer._hide_overlay = True
    self.viewer.cam.trackbodyid = 0
    self.viewer.cam.distance = 1.0 * self.model.stat.extent
    self.viewer.cam.elevation = -90
  
  def reset_model(self):
    while True:
      # Repeat until reaching a stable initial state.
      qpos = self.init_qpos + self.np_random.uniform(low=0.0, high=2*math.pi, size=self.model.nq) 
      qvel = self.init_qvel + self.np_random.uniform(low=0.0, high=0.0, size=self.model.nv)
      self.set_state(qpos, qvel)
      self.step(0)
      vel = self.state_vector()[2:3]
      if np.linalg.norm(vel) == 0:
        # Remove the first flame with black observation.
        self.render()
        break
    
    return self._get_obs()
    
  def _get_obs(self):
    theta = self.sim.data.qpos.flat
    return np.concatenate([
      np.cos(theta),
      np.sin(theta),
      self.sim.data.qvel.flat,
      self.get_body_com("fingertip")
    ])