import MDP
import numpy as np

## Define algorithm
def backwards_induction(mdp, T):
    policy = np.zeros([mdp.n_states, T])
    V = np.zeros([mdp.n_states, T])
    Q = np.zeros([mdp.n_states, mdp.n_actions, T])
    for t in range(T-1, -1, -1):
        for s in range(mdp.n_states):
            for a in range(mdp.n_actions):
                if (t==T-1):
                    Q[s,a,t] = mdp.get_reward(s,a)
                else:
                    Q[s,a,t] = mdp.get_reward(s,a) + sum([V[s2, t+1] * mdp.get_transition_probability(s,a,s2) for s2 in range(mdp.n_states)])
            policy[s,t] = np.argmax(Q[s,:,t])
            V[s,t] = max(Q[s,:,t])
    return policy, V, Q

n_actions = 2
n_states = 2
T = 100
mdp = MDP.DiscreteMDP(n_states, n_actions)

# define transition probabilities
P = np.array([[[0.5, 0.5],
               [0, 1]],
              [[0, 1],
               [0, 1]]])

mdp.set_transition_probability(P)

# set rewards
R = np.array([[5,10],[-1,-1]])
mdp.set_rewards(R)

policy, V, Q = backwards_induction(mdp, T)

for s in range(mdp.n_states):
    for a in range(mdp.n_actions):
        print("S:", s, "A:", a, mdp.get_transition_probabilities(s,a))

for t in range(T):
    print(policy[:,t])
        
for t in range(T):
    print(V[:,t])

    



    
