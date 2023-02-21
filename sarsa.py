#!/usr/bin/env python
# coding: utf-8

# In[89]:


import numpy as np
import gym
import matplotlib.pyplot as plt
import seaborn as sns


# In[115]:


env = gym.make('CliffWalking-v0')
epsilon = 0.1
episodes = 100
max_steps = 100
alpha = 0.1
gamma = 0.99


# In[98]:


Q = np.zeros((env.observation_space.n, env.action_space.n))


# In[99]:


def choose_action(state):
    action = 0
    if np.random.uniform(0,1) < epsilon:
        if type(state) != type(1):
            action = np.argmax(Q[state[0], :])
        else: 
            action = np.argmax(Q[state, :])
    else:
        action = env.action_space.sample()
    return action

def update(state, state2, reward, action, action2):
    if type(state) != type(2):
        predict = Q[state[0], action]
        target = reward + gamma*Q[state2, action2]
        Q[state[0], action] = Q[state[0], action] + alpha * (target - predict)
    else:
        predict = Q[state, action]
        target = reward + gamma*Q[state2, action2]
        Q[state, action] = Q[state, action] + alpha * (target - predict)


# In[120]:


reward = 0
for episode in range(episodes):
    t = 0
    state1 = env.reset()
    action1 = choose_action(state1)
    
    while t < max_steps:
        
        #env.render()
        
        state2, reward, done, info, p = env.step(action1)
        
        action2 = choose_action(state2)
        
        update(state1, state2, reward, action1, action2)
        state1 = state2
        action1 = action2
        
        t += 1
        reward += 1


# In[121]:


#get_ipython().run_line_magic('matplotlib', '')
sns.heatmap(Q, annot=True, cmap='hot')
plt.show()


# In[ ]:




