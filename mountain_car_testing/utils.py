import gym

def env_discretization(discrete_states, env):

    # Define the number of segments or 'discrete_states' the discrete space will have
    # Take the continuous variable space to discrete space
    DISCRETE_OBS_SIZE = [discrete_states] * (len(env.observation_space.high))
    # And now the discrete
    DISCRETE_OBS_WIN_SIZE = (env.observation_space.high - env.observation_space.low) / DISCRETE_OBS_SIZE

    return  DISCRETE_OBS_SIZE,DISCRETE_OBS_WIN_SIZE

def env_test_step(env):
    # To simulate the agent interacting with the environment, a loop that goes through the desired time steps can be setup
    # with multiple 'env.step(action)' and 'env.render()' to update the visual representation
    for _ in range(100):
        observation, reward, _, _ = env.step(2)
        print('Observation is {}'.format(observation))
        print('Reward is {}'.format(reward))
        env.render()
        # sleep(0.05)

def env_test_episodes(env, num_episodes=10):
    # The Mountain Car problem is supposed to be an episodic problem, at each episode's end the policy values and the
    # state values are updated.
    # These episodes are segmented by a terminal state that defines the episode's end.
    # We set a number of episodes and reset the environment
    env.reset()

    for n in range(num_episodes):
        # There is no need to use a 'for' loop this time
        env.reset()
        while True:
            action = env.action_space.sample()
            # For each decision in each time step the following can be implemented
            # - The 'env.action_space.sample()' will return a random action within the action space with equal probability
            # - The 'env.step(input)' takes an input an integer that represents an action
            observation, reward, done, _ = env.step(action)
            print('Observation is {}'.format(observation))
            print('Reward is {}'.format(reward))
            # In RL each action is mapped with a reward, the 'env.step()' returns multiple values, one of these being the reward
            # To visualize the environment, to render the desired env you need to use the following function
            env.render()
            # sleep(0.05)

            # Detect the terminal point of the simulation, if the episode is terminated, then break while and go to
            # the next episode
            if done:
                print("Episode {} ended".format(n))
                break

def get_discrete_state(env,state):
    discrete_state = (state -  env.observation_space.low)
    return tuple(discrete_state.astype(np.int))

def q_learning_greedy(env, learning_rate, discount_rate, episodes,render_rate):
    # Main loop for a greedy Q-learning algorithm
    for episode in range(episodes):

        if episode % render_rate == 0:
            render = True
            print("Episode {}".format(episode))
        else:
            render = False    

        discrete_state = get_discrete_state(env,env.reset())
        done = False

    
        while not done:

            action = np.argmax(q_table[discrete_state])
            observation, reward, done, _ = env.step(action)
        
            if render:
                env.render()

            new_discrete_state = get_discrete_state(env,observation)

            # The gradient descent equivalent in RL
            if not done:
                
                # Get maximum possible Q value from table
                max_future_q = np.max(q_table[new_discrete_state])
                # Get current Q value 
                current_q = q_table[discrete_state + (action,)]
                # Set new Q value
                new_q = (1-learning_rate) * current_q + learning_rate * (reward + discount_rate * max_future_q)
                # Replace Q value
                q_table[discrete_state + (action,)] = new_q

            elif(observation[0] >=  env.goal_position):
                print(f"Goal achieved in episode {episode}")
                q_table[ discrete_state + (action,)] = 0
            
            discrete_state = new_discrete_state

    env.close()
print("-----------------")
print("Simulation's end.")
print("-----------------")



