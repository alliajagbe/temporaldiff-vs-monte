import numpy as np
import matplotlib.pyplot as plt

# sarsa algorithm for cliff walking
def sarsa_cliffwalk(alpha, gamma, epsilon, num_episodes):
    # initialize Q(s,a) arbitrarily
    Q = np.zeros((4,12,4))
    # initialize state
    state = [3,0]
    # initialize action
    action = np.random.randint(0,4)
    # loop for each episode
    for episode in range(num_episodes):
        # loop for each step of episode
        while state != [3,11]:
            # choose A from S using policy derived from Q (e-greedy)
            if np.random.rand() < epsilon:
                action = np.random.randint(0,4)
            else:
                action = np.argmax(Q[state[0],state[1],:])
            # take action A, observe R, S'
            next_state = state[:]
            if action == 0:
                next_state[0] -= 1
            elif action == 1:
                next_state[1] += 1
            elif action == 2:
                next_state[0] += 1
            elif action == 3:
                next_state[1] -= 1
            if next_state[0] < 0 or next_state[0] > 3 or next_state[1] < 0 or next_state[1] > 11:
                next_state = state[:]
            if next_state == [3,11]:
                reward = 0
            elif next_state == [3,0]:
                reward = -100
            else:
                reward = -1
            # choose A' from S' using policy derived from Q (e-greedy)
            if np.random.rand() < epsilon:
                next_action = np.random.randint(0,4)
            else:
                next_action = np.argmax(Q[next_state[0],next_state[1],:])
            # Q(S,A) <- Q(S,A) + alpha[R + gamma*Q(S',A') - Q(S,A)]
            Q[state[0],state[1],action] += alpha*(reward + gamma*Q[next_state[0],next_state[1],next_action] - Q[state[0],state[1],action])
            # S <- S'
            state = next_state[:]
            # A <- A'
            action = next_action
    return Q

# q-learning algorithm for cliff walking
def qlearning_cliffwalk(alpha, gamma, epsilon, num_episodes):
    # initialize Q(s,a) arbitrarily
    Q = np.zeros((4,12,4))
    # initialize state
    state = [3,0]
    # loop for each episode
    for episode in range(num_episodes):
        # loop for each step of episode
        while state != [3,11]:
            # choose A from S using policy derived from Q (e-greedy)
            if np.random.rand() < epsilon:
                action = np.random.randint(0,4)
            else:
                action = np.argmax(Q[state[0],state[1],:])
                print(action)
            # take action A, observe R, S'
            next_state = state[:]
            if action == 0:
                next_state[0] -= 1
            elif action == 1:
                next_state[1] += 1
            elif action == 2:
                next_state[0] += 1
            elif action == 3:
                next_state[1] -= 1
            if next_state[0] < 0 or next_state[0] > 3 or next_state[1] < 0 or next_state[1] > 11:
                next_state = state[:]
            if next_state == [3,11]:
                reward = 0
            elif next_state == [3,0]:
                reward = -100
            else:
                reward = -1
            # Q(S,A) <- Q(S,A) + alpha[R + gamma*max_a Q(S',a) - Q(S,A)]
            Q[state[0],state[1],action] += alpha*(reward + gamma*np.max(Q[next_state[0],next_state[1],:]) - Q[state[0],state[1],action])
            # S <- S'
            state = next_state[:]
    return Q


q_sarsa = sarsa_cliffwalk(0.5, 1, 0.1, 500)
q_qlearning = qlearning_cliffwalk(0.5, 1, 0.1, 500)

# plt.imshow(q_qlearning, cmap = 'autumn' , interpolation = 'nearest')
# plt.show()