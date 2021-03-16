import numpy as np

class ExpectedSarsaAgent():
    def agent_init(self, agent_init_info):
        """Setup for the agent called when the experiment first starts.
        
        Args:
        agent_init_info (dict), the parameters used to initialize the agent. The dictionary contains:
        {
            num_states (int): The number of states,
            num_actions (int): The number of actions,
            epsilon (float): The epsilon parameter for exploration,
            step_size (float): The step-size,
            discount (float): The discount factor,
        }
        
        """
        # Store the parameters provided in agent_init_info.
        self.num_actions = agent_init_info["num_actions"]
        self.num_states = agent_init_info["num_states"]
        self.epsilon = agent_init_info["epsilon"]
        self.step_size = agent_init_info["step_size"]
        self.discount = agent_init_info["discount"]
        self.rand_generator = np.random.RandomState(agent_init_info["seed"])
        self.prev_state = (0,0)
        self.prev_action = (0,0)

        # Create an array for action-value estimates and initialize it to zero.
        self.q = np.zeros((self.num_states, self.num_actions)) # The array of action-value estimates.

        
    def agent_start(self, observation):
        """The first method called when the episode starts, called after
        the environment starts.
        Args:
            observation (int): the state observation from the
                environment's evn_start function.
        Returns:
            action (int): the first action the agent takes.
        """
        
        # Choose action using epsilon greedy.
        state = observation
        current_q = self.q[state, :]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        self.prev_state = state
        self.prev_action = action
        return action
    
    def agent_step(self, reward, observation):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            observation (int): the state observation from the
                environment's step based on where the agent ended up after the
                last step.
        Returns:
            action (int): the action the agent is taking.
        """
        
        # Choose action using epsilon greedy.
        state = observation
        current_q = self.q[state,:]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        
     # Perform an update
        # --------------------------
        # your code here
        
        # Policy distribuition vector
        pi = np.zeros(current_q.shape)
        
        for i in range(self.num_actions):
                if( i == np.argmax(current_q)):
                    pi[i] = (1 - self.epsilon) + (self.epsilon/(self.num_actions))
                else:
                    pi[i] = self.epsilon/(self.num_actions)
        
        # Calculate expected Q-value for current state
        expected_q=np.sum(pi*current_q)
        print(expected_q)
        # Last Q-value
        last_q = self.q[self.prev_state,self.prev_action]
        
        # Apply step size
        q_step = self.step_size*(reward + self.discount*expected_q - last_q)
        
        # Update previus Q value
        self.q[self.prev_state,self.prev_action] = last_q + q_step 
        
        #print(self.q)
        # --------------------------
        
        self.prev_state = state
        self.prev_action = action
        return action
        
        self.prev_state = state
        self.prev_action = action
        return action
    
    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        # Perform the last update in the episode
        # --------------------------
        # your code here
        
        # Create last_q var for the sake of brevity
        last_q = self.q[self.prev_state, self.prev_action]
        # Update Q-value
        self.q[self.prev_state, self.prev_action] = last_q + self.step_size*(reward- last_q)

        print(self.q[self.prev_state,:])
        # --------------------------
        
    def argmax(self, q_values):
        """argmax with random tie-breaking
        Args:
            q_values (Numpy array): the array of action-values
        Returns:
            action (int): an action with the highest value
        """
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return self.rand_generator.choice(ties)

    

## -------------------------------------------
#   MAIN CODE
## -------------------------------------------

s = 3
a = 4

epsilon = 0.3

q = np.zeros((s,a))
pi = np.zeros((s,a))

q[2,1] = 1

current_s = 2

current_q = q[current_s,:]
print(np.zeros(current_q.shape))
#print("q matrix")
#print(q)

# Update pi matrix based on epsilon
for i in range(a):
    #print(q[current_s,i])
    if( i == np.argmax(q[current_s,:]) ):
        pi[current_s,i] = (1 - epsilon) + (epsilon/(a))
    else:
        pi[current_s,i] = epsilon/(a)




#print("pi matrix")
print(pi)
#print("state-action pairs of current state")
#print(q[current_s,:])
