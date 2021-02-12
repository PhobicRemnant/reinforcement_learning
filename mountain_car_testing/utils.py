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
            observation, reward, done, _ = env.step(action)
            print('Observation is {}'.format(observation))
            print('Reward is {}'.format(reward))
            env.render()
            # sleep(0.05)

            # Detect the terminal point of the simulation, if the episode is terminated, then break while and go to
            # the next episode
            if done:
                print("Episode {} ended".format(n))
                break