import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

NUM = 5
file_name = 'data/DQN_0.01_0.999_32_{}.csv'

steps = pd.DataFrame()
rewards = pd.DataFrame()

for i in range(NUM):
    df_ = pd.read_csv(file_name.format(i), index_col=False)
    steps = pd.concat([steps, df_.iloc[:, 0]], axis = 1)
    rewards = pd.concat([rewards, df_.iloc[:, 1]], axis=1)

step_mean = steps.mean(1)
step_std = steps.std(1)
reward_mean = rewards.mean(1)
reward_std = rewards.std(1)

plt.title('Learning curve')
plt.xlabel('episodes')
plt.ylabel('steps')
plt.grid()
plt.plot(step_mean, color='red', alpha=1, label='Fit')
plt.fill_between(np.arange(200), step_mean-step_std, step_mean+step_std, color='#539caf', alpha=0.6)
plt.show()

plt.title('Learning curve')
plt.xlabel('episodes')
plt.ylabel('rewards')
plt.grid()
plt.plot(reward_mean, color='red', alpha=1, label='Fit')
plt.fill_between(np.arange(200), reward_mean-reward_std, reward_mean+reward_std, color='#539caf', alpha=0.6)
plt.show()