import gym
import pandas as pd
import argparse

from config import MEMORY_CAPACITY, TEST_EPISODE_NUM, BATCH_SIZE, E_GREEDY
from brain import DQN

def newReward(obsesrvation, obsesrvation_):
    return abs(obsesrvation_[0] - (-0.5))

def test(mothod, model_path):
    #load model
    RL.load_model(model_path)

    steps, rewards = [], []
    for episode in range(TEST_EPISODE_NUM):
        # initial
        observation = env.reset()

        if method == 'SARSA':
            action = RL.choose_action(observation)

        iter_cnt, total_reward = 0, 0
        while True:
            iter_cnt += 1

            # fresh env
            env.render()

            if method == 'DQN':
                # RL choose action based on observation
                action = RL.choose_action(observation)
                # RL take action and get next observation and reward
                observation_, reward, done, _ = env.step(action)
                reward = newReward(observation, observation_)

            # accumulate reward
            total_reward += reward
            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                total_reward = round(total_reward, 2)
                steps.append(iter_cnt)
                rewards.append(total_reward)
                print("Episode {} finished after {} timesteps, total reward is {}".format(episode + 1, iter_cnt, total_reward))
                break

    # end of game
    print('--------------------------------')
    print('After {} episode,\nthe average step is {},\nthe average reward is {}.'.format(TEST_EPISODE_NUM, sum(steps)/len(steps), sum(rewards)/len(rewards)))
    env.close()

if __name__ == "__main__":

    # argument
    parse = argparse.ArgumentParser()
    parse.add_argument('-m', '--method',
                        default='DQN',
                        help='Choose which rl algorithm used (DQN)')
    parse.add_argument('-t', '--test',
                        default='model/DQN/eval_0.01_{}_{}.pkl'.format(E_GREEDY, BATCH_SIZE),
                        help='The test model path')
    args = parse.parse_args()

    # env setup
    env = gym.make('MountainCar-v0')
    env = env.unwrapped

    # algorithm setup
    method = args.method
    if method == 'DQN':
        print("Use DQN...")
        print('--------------------------------')
        env_shape = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape  # to confirm the shape
        RL = DQN(action_n=env.action_space.n, state_n=env.observation_space.shape[0], env_shape=env_shape)
    else:
        print("Error method! Use DQN instead.")
        print('--------------------------------')
        env_shape = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape  # to confirm the shape
        RL = DQN(action_n=env.action_space.n, state_n=env.observation_space.shape[0], env_shape=env_shape)

    test(method, args.test)