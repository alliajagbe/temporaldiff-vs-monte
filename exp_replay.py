#%%

import gym
import numpy as np
from collections import deque

#%%

# Initialize the environment
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Initialize the Q-table and replay buffer
q_table = np.random.rand(state_size, action_size)
replay_buffer = deque(maxlen=10000)

# Set the hyperparameters
gamma = 0.95
learning_rate = 0.1
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
batch_size = 32
episodes = 1000

#%%

# Define the epsilon-greedy policy
def epsilon_greedy_policy(state):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state])
    
#%%

# Loop over episodes
for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Choose an action using the epsilon-greedy policy
        action = epsilon_greedy_policy(state)
        
        # Take the action and observe the reward and the next state
        next_state, reward, done, info1, info2 = env.step(action)
        
        # Store the experience in the replay buffer
        replay_buffer.append((state, action, reward, next_state, done))
        
        # Update the Q-table using experience replay
        if len(replay_buffer) >= batch_size:
            # Sample a batch of experiences from the replay buffer
            batch = np.array(replay_buffer)[np.random.choice(len(replay_buffer), size=batch_size, replace=False)]
            states, actions, rewards, next_states, dones = batch.T
            
            # Compute the target Q-values for the batch
            q_targets = rewards + gamma * np.max(q_table[next_states], axis=1) * (1 - dones)
            
            # Update the Q-values in the Q-table
            q_table[states, actions] = (1 - learning_rate) * q_table[states, actions] + learning_rate * q_targets
        
        # Update the state and total reward
        state = next_state
        total_reward += reward
    
    # Decay epsilon
    if epsilon > min_epsilon:
        epsilon *= epsilon_decay
    
    # Print the total reward for the episode
    print("Episode: {}, total reward: {}".format(episode, total_reward))
# %%
