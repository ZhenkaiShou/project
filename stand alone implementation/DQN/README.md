# Simple implementation of Deep Q-Network
## Basics
This repository implements the Deep Q-Network (DQN) algorithm with 3 different vairants: 
- standard
- asynchronous
- parallel
## Training Pipeline of Different Variations
### Standard
- initialize network variables
- initialize the environment
- loop until reaching maximum steps
  - sample action using exploration policy
  - store data into replay buffer
  - sample training data from replay buffer
  - train the network for one mini-batch
### Asynchronous
- initialize the global network variables
- create N processes
- for each process do the following
  - initialize a local environment
  - loop until reaching maximum global steps
    - get a copy of the latest global network
    - sample action using exploration policy
    - store data into replay buffer
    - sample training data from replay buffer
    - compute the gradients of the local network based on the training data
    - apply the gradients to the global network
    
### Parallel

## Performance
To save time, the agent is trained on the atari game "Pong" since this is a simple environment which can be easily solved by exploiting the weakness of the computer-controlled opponent.
### Learning Curves
The learning curve w.r.t. the episodic reward during training is shown below. From left to right: standard, asynchornous, parallel.

<p float="center">
  <img src="/stand%20alone%20implementation/DQN/Standard/Figures/Training/dqn.png" width="30%"/>
  <img src="/stand%20alone%20implementation/DQN/Asynchronous/Figures/Training/async_dqn.png" width="30%"/>
  <img src="/stand%20alone%20implementation/DQN/Parallel%20Environments/Figures/Training/par_dqn.png" width="30%"/>
</p>

### Training Time
All these 3 variants are trained via Amazon Web Service with a g3.4xlarge instance. For asynchronous and parallel implementation, 10 workers / environments are created.

The training time for each variant is shown below.

|    Variant   | Time(s) |
|:------------:|:-------:|
|   Standard   |  12812  |
| Asynchronous |   5523  |
|   Parallel   |  12113  |

## Reference

- [Deepmind DQN](https://deepmind.com/research/dqn/)

