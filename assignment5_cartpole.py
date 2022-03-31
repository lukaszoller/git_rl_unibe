import gym
import numpy as np
from tensorflow import keras
from keras import layers
from tensorflow.keras.optimizers import Adam
from rl.agents.sarsa import SARSAAgent
from rl.policy import EpsGreedyQPolicy
import matplotlib.pyplot as plt


###### code from https://medium.com/swlh/learning-with-deep-sarsa-openai-gym-c9a470d027a

#Setting up the environment

env = gym.make('CartPole-v1')
seed_val = 456
env.seed(seed_val)
np.random.seed(seed_val)

#Getting the state and action space

states = env.observation_space.shape[0]
actions = env.action_space.n


# Defining a Neural Network function for our Cartpole agent

def agent(states, actions):
    """Creating a simple Deep Neural Network."""

    model = keras.Sequential()
    model.add(layers.Flatten(input_shape=(1, states)))
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(actions, activation='linear'))
    return model

#Getting our neural network
model = agent(states, actions)

#Defining SARSA Keras-RL agent: inputing the policy and the model
sarsa = SARSAAgent(model=model, nb_actions=actions, policy=EpsGreedyQPolicy())

#Compiling SARSA with mean squared error loss
sarsa.compile(Adam(lr=0.001), metrics=["mse"])

#Training the agent for 50000 steps
sarsa.fit(env, nb_steps=50, visualize=False, verbose=1)


#Visualizing our resulted rewards

plt.plot(scores.history['episode_reward'])
plt.xlabel('Episode')
plt.ylabel('Testing total reward')
plt.title('Total rewards over all episodes in testing')
plt.show()

