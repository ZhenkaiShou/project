# Implementation of World Models
## Basics
This repository implements the [Large-Scale Study of Curiosity-Driven Learning](https://pathak22.github.io/large-scale-curiosity/resources/largeScaleCuriosity2018.pdf) in the Atari environment.

An example of an agent playing Breakout with pure curiosity (the reward is only used for measuring the performance, **not for training**):
<p align="center">
  <img src="/paper%20reproduction/Large-Scale%20Study%20of%20Curiosity-Driven%20Learning/Figures/gameplay.gif" />
</p>

## Dependencies
- tensorflow
- gym
- atari-py

## Usage

To change settings:
- For global settings (environment, feature learning method, ...), please modify `config.py`,
- For other settings (batch size, number of environments, ...), please modify the corresponding file.

To train an agent:
```
python training.py
```

To view the agent playing games,
```
python test.py
```

## Reference
- [Large-Scale Study of Curiosity-Driven Learning](https://pathak22.github.io/large-scale-curiosity/resources/largeScaleCuriosity2018.pdf)
- [Official webpage](https://pathak22.github.io/large-scale-curiosity/)
