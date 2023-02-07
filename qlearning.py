import numpy as np

# defining the constant parameters
n_states = 6
actions = 4
n_episodes = 1000
alpha = 0.1
gamma = 0.99
epsilon = 0.1

# initializing the required parameters
P = np.zeros((n_states, actions, n_states))
R = np.zeros((n_states, actions, n_states))
Q = np.zeros((n_states, actions))

# setting the terminal states
# state 0 as the "pit"
# and state (n_states - 1) as "the goal"
terminal_states = [0, n_states - 1]

# setting our reward values
# 100 if it gets to the goal
# -100 if it slips into the pit
R[:,:,n_states-1] = 100
R[0:,:] = -100


# running the algorithm for all episodes
for episode in range(n_episodes):

    # identifying all non-terminal states
    non_terminal_states = [i for i in range(n_states) if i not in terminal_states]

    # because we are just starting, we choose a state randomly to explore
    state = np.random.choice(non_terminal_states)

    # running for each step in an episode
    while state not in terminal_states:

        # applying the behaviour policy (eps-greedy) to decide whether we exploit or explore
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice(actions)
        else:
            action = np.argmax(Q[state, :])


        next_state_probs = P[state, action, :]
        prob_sum = np.sum(next_state_probs)
        if prob_sum == 0:
            next_state_probs = np.ones(n_states) / n_states
        else:
            next_state_probs = next_state_probs / prob_sum


        next_state = np.random.choice(n_states, p=next_state_probs)

        # getting the reward for each state, action, and next_state trio
        reward = R[state, action, next_state]

        # updating our Q-table accordingly with the required parameters on each step
        # by applying the Q-learning formula
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])

        state = next_state

# now, we use the Q-table generated to make decisions
non_terminal_states = [i for i in range(n_states) if i not in terminal_states]
state = np.random.choice(non_terminal_states)
while state not in terminal_states:
    action = np.argmax(Q[state, :])
    next_state_probs = P[state, action, :]
    prob_sum = np.sum(next_state_probs)
    if prob_sum == 0:
        next_state_probs = np.ones(n_states) / n_states
    else:
        next_state_probs = next_state_probs / prob_sum
    next_state = np.random.choice(n_states, p=next_state_probs)

    print(f"From your current state {state},")
    print(f"Take action {action} to go to the next state {next_state}.")
    print("\nYou're welcome :)")
    state = next_state




# import matplotlib.pyplot as plt

# plt.imshow(Q, cmap='hot', interpolation='nearest')
# plt.xlabel('Action')
# plt.ylabel('State')
# plt.title('Q-table for the States and Respective Actions')
# plt.show()
