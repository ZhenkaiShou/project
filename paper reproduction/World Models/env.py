import numpy as np

from gym.envs.box2d.car_racing import CarRacing
from scipy.misc import imresize

# To disable the logger
# Comment line 216 and 289 in gym.envs.box2d.car_racing.py

def process_obs(image):
  # Remove the status bar (row 84-95).
  obs = image[0:84, :, :]
  # Resize the image to 64 * 64 * 3.
  obs = np.array(imresize(obs, (64, 64)), dtype = np.uint8)
  
  return obs

class MyCarRacing(CarRacing):
  def __init__(self, seed = None):
    super(MyCarRacing, self).__init__()
    self.seed(seed)
  
  def _step(self, action):
    obs, reward, done, _ = super(MyCarRacing, self)._step(action)
    obs = process_obs(obs)
    
    return obs, reward, done, None
  
if __name__=="__main__":
  from pyglet.window import key
  a = np.array( [0.0, 0.0, 0.0] )
  def key_press(k, mod):
    global restart
    if k==0xff0d: restart = True
    if k==key.LEFT:  a[0] = -1.0
    if k==key.RIGHT: a[0] = +1.0
    if k==key.UP:    a[1] = +1.0
    if k==key.DOWN:  a[2] = +0.8   # set 1.0 for wheels to block to zero rotation
  def key_release(k, mod):
    if k==key.LEFT  and a[0]==-1.0: a[0] = 0
    if k==key.RIGHT and a[0]==+1.0: a[0] = 0
    if k==key.UP:    a[1] = 0
    if k==key.DOWN:  a[2] = 0
  env = MyCarRacing()
  env.render()
  env.viewer.window.on_key_press = key_press
  env.viewer.window.on_key_release = key_release
  while True:
    env.reset()
    total_reward = 0.0
    steps = 0
    restart = False
    while True:
      s, r, done, info = env.step(a)
      total_reward += r
      if steps % 200 == 0 or done:
        print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
        print("step {} total_reward {:+0.2f}".format(steps, total_reward))
      steps += 1
      env.render()
      if done or restart: break
  env.monitor.close()