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

