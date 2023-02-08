import numpy as np

# defining the constant parameters
states = 6
actions = 4
episodes = 1000
alpha = 0.1
gamma = 0.99
epsilon = 0.1

# initializing the required parameters
P = np.zeros((states, actions, states))
R = np.zeros((states, actions, states))
Q = np.zeros((states, actions))

# setting the terminal states
# state 0 as the "pit"
# and state (states - 1) as "the goal"
terminal_states = [0, states - 1]

# setting our reward values
# 100 if it gets to the goal
# -100 if it slips into the pit
R[:,:,states-1] = 100
R[0:,:] = -100


# running the algorithm for all episodes
for episode in range(episodes):

    # identifying all non-terminal states
    non_terminal_states = [i for i in range(states) if i not in terminal_states]

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
            next_state_probs = np.ones(states) / states
        else:
            next_state_probs = next_state_probs / prob_sum


        next_state = np.random.choice(states, p=next_state_probs)

        # getting the reward for each state, action, and next_state trio
        reward = R[state, action, next_state]

        # updating our Q-table accordingly with the required parameters on each step
        # by applying the Q-learning formula
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])

        state = next_state










# now, we use the Q-table generated to make decisions
non_terminal_states = [i for i in range(states) if i not in terminal_states]
state = np.random.choice(non_terminal_states)
while state not in terminal_states:
    action = np.argmax(Q[state, :])
    next_state_probs = P[state, action, :]
    prob_sum = np.sum(next_state_probs)
    if prob_sum == 0:
        next_state_probs = np.ones(states) / states
    else:
        next_state_probs = next_state_probs / prob_sum
    next_state = np.random.choice(states, p=next_state_probs)

    if action == 0:
        action = "Up"
    elif action == 1:
        action = "Down"
    elif action == 2:
        action = "Right"
    else:
        action = "Left"

    print(f"From your current state {state},")
    print(f"Go {action} to go to the next state {next_state}.")
    print("\nYou're welcome :)\n")
    state = next_state




# import matplotlib.pyplot as plt

# plt.imshow(Q, cmap='Blues', interpolation='nearest')
# plt.xlabel('Action')
# plt.xticks([0,1,2,3])
# plt.ylabel('State')
# plt.title('Q-table for the States and Respective Actions')
# plt.show()
