# Implementation of Deep Q-Network
## Basics
This repository implements the Deep Q-Network (DQN) algorithm with 3 different vairants: 
- [Standard](https://github.com/ZhenkaiShou/project/tree/master/paper%20reproduction/DQN/Standard)
- [Asynchronous](https://github.com/ZhenkaiShou/project/tree/master/paper%20reproduction/DQN/Asynchronous)
- [Parallel](https://github.com/ZhenkaiShou/project/tree/master/paper%20reproduction/DQN/Parallel)

## Dependencies
tensorflow
gym
atari-py

## Training Pipeline of Different Variants
### Standard
- Initialize network variables;
- Initialize the environment;
- Initialize the replay buffer;
- Loop until reaching maximum steps:
  - Sample an action using exploration policy;
  - Simulate the environemnt with the sampled action;
  - Store data into the replay buffer;
  - Sample training data from the replay buffer;
  - Train the network for one mini-batch.
### Asynchronous
- Initialize the global network variables;
- Create N processes;
- For each process do the following **simultaneously**:
  - Initialize a local environment;
  - Initialize a local replay buffer;
  - Loop until reaching maximum global steps:
    - Get a copy of the latest global network;
    - Sample an action using exploration policy;
    - Simulate the local environment with the sampled action;
    - Store data into the local replay buffer;
    - Sample training data from the local replay buffer;
    - Compute the gradients of the copied network based on the training data;
    - Apply the gradients to the global network.
### Parallel
- Initialize the network variables;
- Create N processes;
- For each process:
  - Initialize a local environment;
- Initialize the replay buffer;
- Loop until reaching maximum steps:
  - For each process do the following **simultaneously**:
    - Sample an action using exploration policy;
    - Simulate the local environment with the sampled action;
  - Store all data into the replay buffer;
  - Sample training data from the replay buffer;
  - Train the network for one mini-batch.

## Performance
The agent is trained to play the Atari game "Pong" since this is a simple environment which can be easily solved by exploiting the weakness of the computer-controlled opponent.
### Learning Curves
The learning curve w.r.t. the episodic reward during training is shown below. From left to right: standard, asynchornous, parallel.

<p float="center">
  <img src="/paper%20reproduction/DQN/Standard/Figures/Training/dqn.png" width="30%"/>
  <img src="/paper%20reproduction/DQN/Asynchronous/Figures/Training/async_dqn.png" width="30%"/>
  <img src="/paper%20reproduction/DQN/Parallel/Figures/Training/par_dqn.png" width="30%"/>
</p>

### Training Time
All these 3 variants are trained via Amazon Web Service with a g3.4xlarge instance. For asynchronous and parallel implementation, 10 workers / environments are created.

The training time for each variant is shown below.

|    Variant   | Time(s) |
|:------------:|:-------:|
|   Standard   |  12812  |
| Asynchronous |   5523  |
|   Parallel   |  12113  |

## Extra: Attention Visualization
Here shows the attention of the deep neural network when playing the game Pong:
- Expectation: which part of the observation should be focused on.
- Advantage: which part of the observation plays an important role in decision making.

<p float="center">
  <img src="/paper%20reproduction/DQN/Standard/Figures/Visualization/attention.gif" width="50%"/>
</p>

## Reference

- [Deepmind DQN](https://deepmind.com/research/dqn/)

