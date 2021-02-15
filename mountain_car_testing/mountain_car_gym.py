import gym
from utils import *
from time import sleep
import numpy as np
from q_learning_solvers import Qlearning_greedy

env = gym.make('MountainCar-v0')

# From continuous to discrete
DISCRETE_OBS_SIZE, DISCRETE_OBS_WIN_SIZE = env_discretization(20, env)
render_rate = 2000  
# Q-learning paramaters
learning_rate = 0.1    
discount_rate = 0.95   
episodes = 26000

# Create Q-table
q_table = np.random.uniform(high=2,low=-2,size=(DISCRETE_OBS_SIZE + [env.action_space.n]))

# Run Greedy Q-learning 
rl_env = Qlearning_greedy(env,learning_rate, discount_rate)
rl_env.discretize_env(30)
rl_env.solve_env(episodes,render_rate)
