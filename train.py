import gym
import pandas as pd
import argparse

from config import MEMORY_CAPACITY, TRAIN_EPISODE_NUM, BATCH_SIZE, E_GREEDY
from brain import QLearningTable, SarsaTable, DQN

def newReward(obsesrvation, obsesrvation_):
    return abs(obsesrvation_[0] - (-0.5))

def update(method):
    records = []
    for episode in range(TRAIN_EPISODE_NUM):
        # initial
        observation = env.reset()

        if method == 'SARSA':
            action = RL.choose_action(observation)

        iter_cnt, total_reward = 0, 0
        while True:
            iter_cnt += 1

            # fresh env
            env.render()

            if method == 'QL':
                # RL choose action based on observation
                action = RL.choose_action(observation)
                # RL take action and get next observation and reward
                observation_, reward, done, _ = env.step(action)
                reward = newReward(observation, observation_)
                # RL learn from this transition
                RL.learn(str(observation), action, reward, str(observation_))
            elif method == 'SARSA':
                # RL take action and get next observation and reward
                observation_, reward, done, _ = env.step(action)
                reward = newReward(observation, observation_)
                # RL choose action based on observation
                action_ = RL.choose_action(observation)
                # learn from trasition (s, a, r, s, a)
                RL.learn(str(observation), action, reward, str(observation_), action_)
                # swap action
                action = action_
            elif method == 'DQN':
                # RL choose action based on observation
                action = RL.choose_action(observation)
                # RL take action and get next observation and reward
                observation_, reward, done, _ = env.step(action)
                reward = newReward(observation, observation_)
                # RL learn from this transition
                RL.store_transition(observation, action, reward, observation_)
                if RL.memory_counter > MEMORY_CAPACITY:
                    RL.learn()

            # accumulate reward
            total_reward += reward
            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                total_reward = round(total_reward, 2)
                records.append((iter_cnt, total_reward))
                print("Episode {} finished after {} timesteps, total reward is {}".format(episode + 1, iter_cnt, total_reward))
                break

    # end of game
    print('--------------------------------')
    print('game over')
    env.close()

    # save model
    if method == 'DQN':
        RL.save_model()
        print("save model")

    df = pd.DataFrame(records, columns=["iters", "reward"])
    df.to_csv("data/{}_{}_{}_{}.csv".format(method, RL.lr, E_GREEDY, BATCH_SIZE), index=False)

if __name__ == "__main__":

    # argument
    parse = argparse.ArgumentParser()
    parse.add_argument('-m', '--method',
                        default='DQN',
                        help='Choose which rl algorithm used (QL, SARSA, or DQN)')
    parse.add_argument('-lr', '--learning_rate',
                        type=float, default=0.01,
                        help='Learning rate')
    parse.add_argument('-rd', '--reward_decay',
                        type=float, default=0.9,
                        help='Reward decay')
    args = parse.parse_args()

    # env setup
    env = gym.make('MountainCar-v0')
    env = env.unwrapped

    # algorithm setup
    method = args.method
    if method == 'QL':
        print("Use Q-Learning...")
        print('--------------------------------')
        RL = QLearningTable(actions=list(range(env.action_space.n)),
                            learning_rate=args.learning_rate, reward_decay=args.reward_decay)
    elif method == 'SARSA':
        print("Use SARSA...")
        print('--------------------------------')
        RL = SarsaTable(actions=list(range(env.action_space.n)),
                        learning_rate=args.learning_rate, reward_decay=args.reward_decay)
    elif method == 'DQN':
        print("Use DQN...")
        print('--------------------------------')
        env_shape = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape  # to confirm the shape
        RL = DQN(action_n=env.action_space.n, state_n=env.observation_space.shape[0], env_shape=env_shape,
                learning_rate=args.learning_rate, reward_decay=args.reward_decay)
    else:
        print("Error method! Use DQN instead.")
        print('--------------------------------')
        env_shape = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape  # to confirm the shape
        RL = DQN(action_n=env.action_space.n, state_n=env.observation_space.shape[0], env_shape=env_shape,
                learning_rate=args.learning_rate, reward_decay=args.reward_decay)

    update(method)