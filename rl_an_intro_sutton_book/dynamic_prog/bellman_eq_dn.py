import numpy as np

"""
These are a few fucntions that correspond to the Bellman equation on chapter 4 from the book
The course of the University of Alberta has this exercise.
Can be use to understand better the computation of the DP.
"""
def evaluate_policy(env, V, pi, gamma, theta):
    delta = float('inf')
    while delta > theta:
        delta = 0
        for s in env.S:
            v = V[s]
            bellman_update(env, V, pi, s, gamma)
            delta = max(delta, abs(v - V[s]))

    return V

def bellman_update(env, V, pi, s, gamma):
    """Mutate ``V`` according to the Bellman update equation."""
    # YOUR CODE HERE
    res = 0
    for a in env.A:
        pi_s_a = pi[s,a]
        r = env.transitions(s,a)[:,0]
        p = env.transitions(s,a)[:,1]
        res += pi_s_a * (p *(r + gamma*V) ).sum()
    V[s] = res

def improve_policy(env, V, pi, gamma):
    policy_stable = True
    for s in env.S:
        old = pi[s].copy()
        q_greedify_policy(env, V, pi, s, gamma)
        
        if not np.array_equal(pi[s], old):
            policy_stable = False
            
    return pi, policy_stable

def policy_iteration(env, gamma, theta):
    V = np.zeros(len(env.S))
    pi = np.ones((len(env.S), len(env.A))) / len(env.A)
    policy_stable = False
    
    while not policy_stable:
        V = evaluate_policy(env, V, pi, gamma, theta)
        pi, policy_stable = improve_policy(env, V, pi, gamma)
        
    return V, pi

def q_greedify_policy(env, V, pi, s, gamma):
    """Mutate ``pi`` to be greedy with respect to the q-values induced by ``V``."""
    # YOUR CODE HERE
    
    max_actions = len(env.A)
    G = np.zeros(max_actions)
    
    # Action space loop
    for a in env.A:
        #G += (p *(r + gamma*V) ).sum()
        # States space loop 
        for new_state in env.S:
            r = env.transitions(s,a)[new_state,0]
            p = env.transitions(s,a)[new_state,1]
            G[a] += p*(r+gamma*V[new_state])
            
    
    # Set action values to 0 for this state
    pi[s, :] = 0
    # For the max return action set value to 1
    pi[s, np.argmax(G)] = 1

def value_iteration(env, gamma, theta):
    V = np.zeros(len(env.S))
    while True:
        delta = 0
        for s in env.S:
            v = V[s]
            bellman_optimality_update(env, V, s, gamma)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    pi = np.ones((len(env.S), len(env.A))) / len(env.A)
    for s in env.S:
        q_greedify_policy(env, V, pi, s, gamma)
    return V, pi

def bellman_optimality_update(env, V, s, gamma):
    """Mutate ``V`` according to the Bellman optimality update equation."""
    # YOUR CODE HERE
    
    v = np.zeros(V.shape)
    
    # Action space loop
    for a in env.A:
        
        # States space loop 
        for new_state in env.S:
            r = env.transitions(s,a)[new_state,0]
            p = env.transitions(s,a)[new_state,1]
            v[a] += p*(r+gamma*V[new_state])
            
    V[s] = np.max(v)

# ---------------------------------------------
# init
# ---------------------------------------------

n_states = 3
n_actions = 3

V = np.zeros( n_states + 1)
pi = np.ones((n_states + 1, n_actions)) / n_actions
gamma = 0.9

print(V)
print()
print(pi[0])