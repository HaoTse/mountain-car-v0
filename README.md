# MountainCar-v0

## Requirements

### Enviornment
- ubuntu 16.04
- python 3.6.5
- pytorch 0.4.0
- [MountainCar-v0 in OpenAI gym](https://gym.openai.com/envs/MountainCar-v0/)

### Prerequisite
```
pip install -r requirements.txt
```

## Goal
- How to use OpenAI gym to do RL research
- How to implement RL algorithms
- How to evaluate your algorithms

## Usage
- training
```
usage: train.py [-h] [-m METHOD] [-lr LEARNING_RATE] [-rd REWARD_DECAY]

optional arguments:
  -h, --help            show this help message and exit
  -m METHOD, --method METHOD
                        Choose which rl algorithm used (QL, SARSA, or DQN)
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        Learning rate
  -rd REWARD_DECAY, --reward_decay REWARD_DECAY
                        Reward decay

```

- testing
```
usage: test.py [-h] [-m METHOD] [-t TEST]

optional arguments:
  -h, --help            show this help message and exit
  -m METHOD, --method METHOD
                        Choose which rl algorithm used (DQN)
  -t TEST, --test TEST  The test model path

```

## Result
> Use `plot.py` to plot the learning curve.
- learning curve (steps)

![leanring curve (steps)](https://github.com/HaoTse/mountain-car-v0/blob/master/img/steps.png)

- learning curve (rewards)

![leanring curve (rewards)](https://github.com/HaoTse/mountain-car-v0/blob/master/img/rewards.png)

- After 10 episode, the average step is 152.9, and the average reward is 58.8.

## Analysis
> See `report.pdf`.
