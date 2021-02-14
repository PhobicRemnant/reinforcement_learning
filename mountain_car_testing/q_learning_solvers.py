import gym
import numpy as np

class Qlearning_greedy:
    """
    The gradient descent equivalent in RL

    Takes an OpenAI environment with a learning rate and discount rate
    """
    
    def __init__(self, environment, learning_rate=0.1, discount_rate=0.99):

        self.env = environment
        self.win_size =  0
        self.obs_size = 0
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.q_table = 0
        self.epsilon = epsilon


    def discretice_env(self, discrete_states):
        # Define the number of segments or 'discrete_states' the discrete space will have
        # Take the continuous variable space to discrete space
        DISCRETE_OBS_SIZE = [discrete_states] * (len(self.env.observation_space.high))
        # And now the discrete
        DISCRETE_OBS_WIN_SIZE = (self.env.observation_space.high - self.env.observation_space.low) / DISCRETE_OBS_SIZE

        self.obs_size = DISCRETE_OBS_SIZE
        self.win_size = DISCRETE_OBS_WIN_SIZE
        self.q_table = np.random.uniform(high=2,low=-2,size=(DISCRETE_OBS_SIZE + [self.env.action_space.n]))


    def get_discrete_state(self, state):
        discrete_state = (state -  self.env.observation_space.low)/win_size
        return tuple(discrete_state.astype(np.int))

    def solve_env(self,episodes, render_rate):
        """
        Solve method for an OpenAI environment using greedy Q-learning. 
        """
        def get_discrete_state(state):
            discrete_state = (state -  self.env.observation_space.low)/self.win_size
            return tuple(discrete_state.astype(np.int))
        
        # Main loop for a greedy Q-learning algorithm
        for episode in range(episodes):

            # Set a render rate to avoid wasting time rendering every episode
            if episode % render_rate == 0:
                render = True
                print("Episode {}".format(episode))
            else:
                render = False    

            # Init the state for the next episode
            discrete_state = get_discrete_state(self.env.reset())
            done = False

            # Episode loop
            while not done:

                # Take highest value action in the current state
                action = np.argmax(self.q_table[discrete_state])
                # Execute the action to obtain a reward
                observation, reward, done, _ = self.env.step(action)

                # Render condition
                if render:
                    self.env.render()

                # Discretice the next step based on the current observation
                new_discrete_state = get_discrete_state(observation)

                # Q table update
                if not done:
                    
                    # Get maximum possible Q value from table
                    max_future_q = np.max(self.q_table[new_discrete_state])
                    # Get current Q value 
                    current_q = self.q_table[discrete_state + (action,)]
                    # Set new Q value
                    new_q = (1-self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_rate * max_future_q)
                    # Replace Q value
                    self.q_table[discrete_state + (action,)] = new_q

                # Goal reached in Nth episode display
                elif(observation[0] >=  self.env.goal_position):
                    print(f"Goal achieved in episode {episode}")
                    self.q_table[ discrete_state + (action,)] = 0
                
                # Pass the next state value to the current state
                discrete_state = new_discrete_state

        # Close the environment's render window
        self.env.close()
    

