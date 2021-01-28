import gym 
import numpy as np
import random 
import math

"""
This practice example is found on the book:
    Python Reinforcement Learning Projects 

You train and agent to control a continuous process in the 
inverted pendulum problem
"""

"""
Func/Class definitions
"""

def select_action(state_value,explore_rate):
    if random.random() < explore_rate:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_value_table[state_value])
    return action

def select_explore_rate(x):
    return max(min_explore_rate, min(0.5, 1.0 - math.log10((x+1)/25)))

def select_learning_rate(x):
    return max(min_learning_rate, min(0.5, 1.0 - math.log10((x+1)/25)))

def bucketize_state_value(state_value):
    bucket_indexes = []
    for i in range(len(state_value)):
        if state_value[i] <= state_value_bounds[i][0]:
            bucket_index = 0
        elif state_value[i] >= state_value_bounds[i][1]:
            bucket_index = no_buckets[i] - 1
        else:
            bound_width = state_value_bounds[i][1] - state_value_bounds[i][0]
            offset = (no_buckets[i]-1)*state_value_bounds[i][0]/bound_width
            scaling = (no_buckets[i]-1)/bound_width
            bucket_index = int(round(scaling*state_value[i] - offset))
        bucket_indexes.append(bucket_index)
    return tuple(bucket_indexes)


"""
Env/Var init stage
"""
env = gym.make('CartPole-v0')

no_buckets = (1,1,3,6)
no_actions = env.action_space.n

state_value_bounds = list(zip(env.observation_space.low,
                env.observation_space.high))
state_value_bounds[1] = [-0.5,0.5]
state_value_bounds[3] = [-math.radians(50),math.radians(50)]

action_index = len(no_buckets)

q_value_table = np.zeros(no_buckets + (no_actions,))

min_explore_rate = 0.01
min_learning_rate = 0.1

max_episodes = 1000
max_time_steps = 250
streat_to_end = 5
solved_time = 199
discount = 0.99
no_streaks = 0

"""
Agent training 
"""

for episode_no in range(max_episodes):
    explore_rate = select_explore_rate(episode_no)
    learning_rate = select_learning_rate(episode_no)

    ob = env.reset()
    
    start_state_value = bucketize_state_value(ob)
    previous_state_value = start_state_value

    for time_step in range(max_time_steps):
        env.render()
        sel_action =select_action(previous_state_value,explore_rate)
        ob, r, completed, _ = env.step(sel_action)
        state_value = bucketize_state_value(ob)
        best_q_value = np.amax(q_value_table[state_value])
        q_value_table[previous_state_value + (sel_action,)] += learning_rate* (r + discount*(best_q_value) - q_value_table[previous_state_value + (sel_action,)])

        print('Episode number : %d' % episode_no)
        print('Time step : %d' % time_step)
        print('Selection action : %d' % sel_action)
        print('Current state : %s' % str(state_value))
        print('Reward obtained : %f' % r)
        print('Best Q value : %f' % best_q_value)
        print('Learning rate : %f' % learning_rate)
        print('Explore rate : %f' % explore_rate)
        print('Streak number : %d' % no_streaks)

        if completed:
            print('Episode %d finished after %f time steps' % (episode_no, time_step))
            if time_step >= solved_time:
                no_streaks += 1
            else:
                no_streaks = 0
            break
        
        previous_state_value = state_value

        if no_streaks > streat_to_end:
            break
