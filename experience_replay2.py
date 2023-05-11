#%%
import random

#%%
# Define the Q-learning function
def q_learning(state, action, reward, next_state, alpha, gamma, q_table):
    # Update Q-value for the current state and action
    q_table[state][action] = (1 - alpha) * q_table[state][action] + alpha * (reward + gamma * max(q_table[next_state]))
    
    return q_table

#%%
# Define the experience replay function
def experience_replay(memory, batch_size, alpha, gamma, q_table):
    # Randomly sample a batch of experiences from memory
    batch = random.sample(memory, batch_size)
    
    # Iterate over the batch and update Q-values using Q-learning
    for state, action, reward, next_state, done in batch:
        if not done:
            q_table = q_learning(state, action, reward, next_state, alpha, gamma, q_table)
        else:
            q_table[state][action] = (1 - alpha) * q_table[state][action] + alpha * reward
    
    return q_table

#%%
# Define the main function
def main():
    # Initialize the Q-table
    q_table = [[0 for i in range(2)] for j in range(3)]
    
    # Initialize the memory
    memory = [(0, 0, 1, 1, False),
              (1, 1, 0, 2, False),
              (2, 0, -1, 1, True),
              (0, 1, 0, 2, False),
              (2, 0, -1, 1, True),
              (1, 0, 1, 0, False)]
    
    # Set the hyperparameters
    alpha = 0.5
    gamma = 0.9
    batch_size = 2
    
    # Train the Q-learning agent using experience replay
    for i in range(100):
        q_table = experience_replay(memory, batch_size, alpha, gamma, q_table)
    
    # Print the final Q-table
    print("Final Q-table:")
    for i,row in enumerate(q_table):
        print("State {}: {}".format(i, row))

if __name__ == '__main__':
    main()

# %%
