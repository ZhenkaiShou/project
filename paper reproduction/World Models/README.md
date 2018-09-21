# Implementation of World Models
## Basics
This repository implements the [World Models](https://arxiv.org/abs/1803.10122) using the car racing environment.
## Dependencies
tensorflow
gym==0.9.4 (**Note**: The latest version of gym breaks this experiment.)
box2d
scipy
## Running on a Server
Below are some tips of running the experiment on a server.

(1) To render the environment on a server. Here we take AWS Deep Learning AMI with Ubuntu 14.04 as example.
  '''
  mkdir ~/Downloads
  mkdir ~/Downloads/nvidia
  cd ~/Downloads/nvidia
  wget http://us.download.nvidia.com/XFree86/Linux-x86_64/396.51/NVIDIA-Linux-x86_64-396.51.run
  wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run
  sudo chmod +x NVIDIA-Linux-x86_64-396.51.run
  sudo chmod +x cuda_9.0.176_384.81_linux-run
  ./cuda_9.0.176_384.81_linux-run -extract=~/Downloads/nvidia/
  sudo apt-get --purge remove nvidia-*
  sudo nvidia-uninstall
  sudo shutdown -r now # You need to restart the server
  sudo ./Downloads/nvidia/NVIDIA-Linux-x86_64-396.51.run --no-opengl-files
  sudo ./Downloads/nvidia/cuda-linux.9.0.176-22781540.run --no-opengl-libs
  '''

(2) To install Box2D on Ubuntu:
  '''
  sudo apt-get install build-essential python-dev swig python-pygame
  pip install Box2D
  '''

(3) To install and render the environment of gym==0.9.4:
  '''
  pip install gym==0.9.4
  sudo apt install xvfb
  xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python random_sampling.py
  '''

## Performance
### Visualization of Controller:
Due to limited resources, the training of the controller module is not fully complete.
<p float="center">
  <img src="/paper%20reproduction/World%20Models/Figures/Controller%20Visualization/0000.gif" />
</p>

## Reference
- [World Models](https://arxiv.org/abs/1803.10122)
- [World Models implementation by Hardmaru](https://github.com/hardmaru/WorldModelsExperiments)

