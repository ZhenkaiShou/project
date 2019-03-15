# Simple implementation of Deep Q-Network
## Basics
This repository implements the Deep Q-Network (DQN) algorithm in 3 different ways: 
- standard implementation
- asynchronous implementation
- parallel environment
## Performance
To save time, the agent is trained on the atari game "Pong" since this is a simple environment which can be easily solved by exploiting the weakness of the computer-controlled opponent.

Below are the learning curve w.r.t. the episodic reward during training.

### Standard implementation:
<p float="center">
  <img src="/stand%20alone%20implementation/DQN/Standard/Figures/Training/dqn.png" width="40%"/>
</p>

### Asynchronous implementation:
<p float="center">
  <img src="/stand%20alone%20implementation/DQN/Asynchronous/Figures/Training/async_dqn.png" width="40%"/>
</p>

### Parallel environment:
<p float="center">
  <img src="/stand%20alone%20implementation/DQN/Parallel%20Environment/Figures/Training/par_dqn.png" width="40%"/>
</p>

## Reference

- [Deepmind DQN](https://deepmind.com/research/dqn/)

