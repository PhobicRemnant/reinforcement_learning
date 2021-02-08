import gym
from utils import *
from time import sleep

env = gym.make('MountainCar-v0')

# Resetting the env returns the observation space values
observation = env.reset()
print(observation)
# The actions are defined as a Discrete type, where each element is a integer
actions = env.action_space
print(actions)
# For each decision in each time step the following can be implemented
# - The 'env.action_space.sample()' will return a random action within the action space with equal probability
# - The 'env.step(input)' takes an input an integer that represents an action
observation, reward, _, _ = env.step(env.action_space.sample())
# In RL each action is mapped with a reward, the 'env.step()' returns multiple values, one of these being the reward
# To visualize the environment, to render the desired env you need to use the following function
env.render()
# Testing the step function
env_test_step(env)
env_test_episodes(env)










env.close()
print("Simulation's end.")