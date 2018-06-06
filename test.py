import gym
from brain import QLearningTable, SarsaTable

def update():
    for episode in range(1000):
        # initial observation
        observation = env.reset()

        # RL choose action based on observation (SARSA)
        action = RL.choose_action(str(observation))

        cnt, total_reward = 0, 0
        while True:
            cnt += 1

            # fresh env
            env.render()

            # RL choose action based on observation (QLearning)
            # action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward
            observation_, reward, done, _ = env.step(action)

            # RL choose action based on next observation (SARSA)
            action_ = RL.choose_action(str(observation_))

            # RL learn from this transition (QLearning)
            # RL.learn(str(observation), action, reward, str(observation_))
            # RL learn from this transition (s, a, r, s, a) (Sarsa)
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # swap observation
            observation = observation_
            action = action_
            total_reward += reward

            # break while loop when end of this episode
            if done:
                print("Episode {} finished after {} timesteps, total reward is {}".format(episode + 1, cnt, total_reward))
                break

    # end of game
    print('game over')
    env.close()

if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    RL = SarsaTable(actions=list(range(env.action_space.n)))

    update()