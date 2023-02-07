import numpy as np

# Define the environment
n_states = 6
n_actions = 4
P = np.zeros((n_states, n_actions, n_states))
R = np.zeros((n_states, n_actions, n_states))
terminal_states = [0, n_states - 1]

# Define the Q-Learning algorithm
n_episodes = 1000
alpha = 0.1
gamma = 0.99
epsilon = 0.1
Q = np.zeros((n_states, n_actions))

for episode in range(n_episodes):
    non_terminal_states = [i for i in range(n_states) if i not in terminal_states]
    state = np.random.choice(non_terminal_states)
    while state not in terminal_states:
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice(n_actions)
        else:
            action = np.argmax(Q[state, :])
        next_state_probs = P[state, action, :]
        prob_sum = np.sum(next_state_probs)
        if prob_sum == 0:
            next_state_probs = np.ones(n_states) / n_states
        else:
            next_state_probs = next_state_probs / prob_sum
        next_state = np.random.choice(n_states, p=next_state_probs)
        reward = R[state, action, next_state]
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[next_state, action])
        state = next_state

# Use the learned Q-table to make decisions
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




import matplotlib.pyplot as plt

plt.imshow(Q, cmap='hot', interpolation='nearest')
plt.xlabel('Action')
plt.ylabel('State')
plt.title('Q-table for the States and Respective Actions')
plt.show()
