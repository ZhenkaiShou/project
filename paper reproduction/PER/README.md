# Implementation of Prioritized Experience Replay
## Basics
This repository implements the Prioritized Experience Replay (PER) algorithm, which is an extension to the Deep Q-Network (DQN). In PER, samples with higher training errors are more likely to be sampled for training.

## Performance
Two variants of agents are trained on the Atari game "Seaquest":
- The first agent `Seaquest_uniform` is trained with uniform sampling strategy
- The second agent `Seaquest_prioritized` is trained with prioritized sampling strategy

Final `ϵ` in ϵ-greedy policy is set to 0.01 to further exploit the different between these two agents. 

<p float="center">
  <img src="/paper%20reproduction/PER/Figures/Training/Seaquest.png" width="60%"/>
</p>


## Reference

- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)


