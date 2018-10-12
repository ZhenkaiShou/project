# Simple Implementation of Deep Deterministic Policy Gradient
## Basics
This repository implements the deep deterministic policy gradient (DDPG) algorithm which can be applied to many reinforcement learning tasks. In this repository we use this algorithm to solve the pendulum game.
## Performance
### Update the Actor by Minimizing the Actor Loss
DDPG can be implemented in such a way that we can update the actor network $ \pi(s) $ by minimizing the actor loss $ l_{\pi}=-1/N*\sum Q(s, a)=-1/N*\sum Q(s, \pi(s)) $.
### Final Reward w.r.t. Training Progress:
<p float="center">
  <img src="/stand%20alone%20implementation/DDPG/Figures/Training/pendulum.png" width="50%"/>
</p>

### Visualization:
<p float="center">
  <img src="/stand%20alone%20implementation/VAE/Figures/Visualization/pendulum.png" width="50%"/>
</p>

## Reference

- [OpenAI DDPG Baseline](https://github.com/openai/baselines/tree/master/baselines/ddpg)

