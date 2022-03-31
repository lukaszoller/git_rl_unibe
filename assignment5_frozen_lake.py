import numpy as np
import gym
import pandas as pd
import matplotlib.pyplot as plt

# code partially from https://www.geeksforgeeks.org/sarsa-reinforcement-learning/


#Function to choose the next action
def choose_action(state, epsilon, Q):
	if np.random.uniform(0, 1) < epsilon:
		action = env.action_space.sample()
	else:
		action = np.argmax(Q[state, :])
	return action

#Function to learn the Q-value
def update(state, state2, reward, action, action2, Q, alpha):
	predict = Q[state, action]
	target = reward + gamma * Q[state2, action2]
	Q[state, action] = Q[state, action] + alpha * (target - predict)

def return_new_reward(done, reward):
    if done and reward == 1:    # goal
        return 10
    elif done and reward == 0:  # hole
        return -10
    else:   # normal step (not hole, not goal)
        return -1


def print_q_policy(Q):
    sidelen_map = int(np.sqrt(len(Q)))
    policy = np.empty(shape=(sidelen_map, sidelen_map), dtype=str)

    for i in range(len(Q)): # loop over all rows in Q
        # get map_row and map_col indices
        map_row = int(np.floor(i/sidelen_map))
        map_col = i%sidelen_map

        # get action from Q
        action_int = np.argmax(Q[i])

        # recode action_int to str
        if action_int == 0:
            policy[map_row, map_col] = "L"
        elif action_int == 1:
            policy[map_row, map_col] = "D"
        elif action_int == 2:
            policy[map_row, map_col] = "R"
        else:
            policy[map_row, map_col] = "U"

        # print X for holes and goal
        if np.sum(Q[i]) == 0:
            policy[map_row, map_col] = "X"

    return  policy

def test_epsilon(epsilon_array, alpha):
    # Initializing the reward
    reward = 0
    eps_test = []

    ## test all epsilons
    for epsilon in epsilon_array:
        # we run n experiments and compute the mean for each episode (mean_episode_reward is stored in eps_test)
        mean_episode_reward = np.zeros(shape=(total_episodes,))

        # repeat the test for each epsilon n times
        for i in range(n):
            # store reward for each episode in episode_rewards
            episode_rewards = []

            # loop over episodes (this is the actual run / the optimization of the q table)

            # Initializing the Q-matrix
            Q = np.zeros((env.observation_space.n, env.action_space.n))
            for episode in range(total_episodes):
                # print(episode)
                t = 0
                state1 = env.reset()
                action1 = choose_action(state1, epsilon, Q)
                episode_reward = 0
                epsilon *= epsilon_decay ** episode
                while t < max_steps:
                    # Getting the next state
                    state2, reward, done, info = env.step(action1)

                    # use my own reward function (function from gym returns only +1 if goal)
                    reward = return_new_reward(done, reward)
                    episode_reward += reward
                    # Choosing the next action
                    action2 = choose_action(state2, epsilon, Q)

                    # Learning the Q-value
                    update(state1, state2, reward, action1, action2, Q, alpha)

                    state1 = state2
                    action1 = action2

                    # Updating the respective vaLues
                    t += 1
                    # reward += 1 # no idea why they increase the reward here

                    # If at the end of learning process
                    if done:
                        break
                # store rewards in data
                episode_rewards.append(episode_reward)

            # compute mean of the reward arrays (stepwise mean function
            # avg = avg + (old - new)*1/n
            mean_episode_reward = np.add(mean_episode_reward,
                                         np.multiply(
                                             np.subtract(episode_rewards, mean_episode_reward),
                                             1 / (i + 1)))

        eps_test.append(mean_episode_reward)

    # Evaluating the performance
    print("Performance : ", reward / total_episodes)

    # Visualizing the Q-matrix
    print(Q)
    print(print_q_policy(Q))

    # plot
    # plt.plot(episode_rewards_eps_test)
    for i in range(len(eps_test)):
        plt.plot(eps_test[i], label=str(epsilon_array[i]),
                 linewidth=0.5)

    plt.legend()
    plt.title(r'Mean reward per episode, differentte $\epsilon$')
    plt.ylabel('mean reward per episode')
    plt.xlabel('episode')
    plt.show()

def test_alpha(alpha_array, epsilon):
    # Initializing the reward
    reward = 0
    eps_test = []

    ## test all epsilons
    for alpha in alpha_array:
        # we run n experiments and compute the mean for each episode (mean_episode_reward is stored in eps_test)
        mean_episode_reward = np.zeros(shape=(total_episodes,))

        # repeat the test for each epsilon n times
        for i in range(n):
            # store reward for each episode in episode_rewards
            episode_rewards = []

            # loop over episodes (this is the actual run / the optimization of the q table)

            # Initializing the Q-matrix
            Q = np.zeros((env.observation_space.n, env.action_space.n))
            for episode in range(total_episodes):
                # print(episode)
                t = 0
                state1 = env.reset()
                action1 = choose_action(state1, epsilon, Q)
                episode_reward = 0
                epsilon *= epsilon_decay ** episode
                while t < max_steps:
                    # Getting the next state
                    state2, reward, done, info = env.step(action1)

                    # use my own reward function (function from gym returns only +1 if goal)
                    reward = return_new_reward(done, reward)
                    episode_reward += reward
                    # Choosing the next action
                    action2 = choose_action(state2, epsilon, Q)

                    # Learning the Q-value
                    update(state1, state2, reward, action1, action2, Q, alpha)

                    state1 = state2
                    action1 = action2

                    # Updating the respective vaLues
                    t += 1
                    # reward += 1 # no idea why they increase the reward here

                    # If at the end of learning process
                    if done:
                        break
                # store rewards in data
                episode_rewards.append(episode_reward)

            # compute mean of the reward arrays (stepwise mean function
            # avg = avg + (old - new)*1/n
            mean_episode_reward = np.add(mean_episode_reward,
                                         np.multiply(
                                             np.subtract(episode_rewards, mean_episode_reward),
                                             1 / (i + 1)))

        eps_test.append(mean_episode_reward)

    # Evaluating the performance
    print("Performance : ", reward / total_episodes)

    # Visualizing the Q-matrix
    print(Q)
    print(print_q_policy(Q))

    # plot
    # plt.plot(episode_rewards_eps_test)
    for i in range(len(eps_test)):
        plt.plot(eps_test[i], label=str(epsilon_array[i]),
                 linewidth=0.5)

    plt.legend()
    plt.title(r'Mean reward per episode, different stepsizes $\alpha$')
    plt.ylabel('mean reward per episode')
    plt.xlabel('episode')
    plt.show()


### run actual tests:
#Building the environment
env = gym.make('FrozenLake8x8-v1', is_slippery = False)
# env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False)
# Env information
    #     "4x4":[
    #         "SFFF",
    #         "FHFH",
    #         "FFFH",
    #         "HFFG"
    #         ]

    # MAP Coding:
    # S = Start
    # F = Frozen Lake
    # H = Hole
    # G = Goal

    # States numbering:
    #     [[ 0  1  2  3]
    #      [ 4  5  6  7]
    #      [ 8  9 10 11]
    #      [12 13 14 15]]

    #
    # ACTIONS:
    # - 0: LEFT
    # - 1: DOWN
    # - 2: RIGHT
    # - 3: UP


#Defining the different parameters
n = 10 # size of the experiments from which mean will be taken
epsilon_array = [0.01, 0.1, 0.2, 0.5, 0.9]
alpha_array = [0.01, 0.1, 0.2, 0.5, 0.9]
epsilon_decay = 0.99
total_episodes = 500
max_steps = 50
alpha = 0.1
gamma = 0.9        # discount rate

#test_epsilon(epsilon_array, alpha=0.1)

test_alpha(alpha_array, epsilon=0.1)