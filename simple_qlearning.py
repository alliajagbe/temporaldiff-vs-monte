#%%

import random

#%%

# Define the Q-learning function
def q_learning(state, action, reward, next_state, alpha, gamma, q_table):
    # Update Q-value for the current state and action
    q_table[state][action] = (1 - alpha) * q_table[state][action] + alpha * (reward + gamma * max(q_table[next_state]))
    
    return q_table

#%%
# Define the main function
def main():
    # Initialize the Q-table
    q_table = [[0 for i in range(2)] for j in range(3)]
    
    # Set the hyperparameters
    alpha = 0.5
    gamma = 0.9
    
    # Train the Q-learning agent
    for i in range(500):
        # Select a random initial state
        state = random.randint(0, 2)
        done = False
        
        while not done:
            # Select an action using an epsilon-greedy policy
            if random.random() < 0.1:
                action = random.randint(0, 1)
            else:
                action = q_table[state].index(max(q_table[state]))
            
            # Simulate the next state and reward based on the selected action
            if state == 0 and action == 0:
                next_state = 1
                reward = 1
                done = False
            elif state == 0 and action == 1:
                next_state = 2
                reward = 0
                done = False
            elif state == 1 and action == 0:
                next_state = 0
                reward = 1
                done = False
            elif state == 1 and action == 1:
                next_state = 2
                reward = -1
                done = False
            elif state == 2 and action == 0:
                next_state = 1
                reward = 0
                done = True
            elif state == 2 and action == 1:
                next_state = 0
                reward = 0
                done = True
                
            # Update Q-values using Q-learning
            q_table = q_learning(state, action, reward, next_state, alpha, gamma, q_table)
            
            # Update the current state
            state = next_state
    
    # Print the final Q-table
    print("Final Q-Table:")
    for i, row in enumerate(q_table):
        print("State {}: {}".format(i, row))

#%% 
if __name__ == '__main__':
    main()

# %%
