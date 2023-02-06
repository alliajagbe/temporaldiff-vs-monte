import numpy as np
import random

state_size = 16
action_size = 4
epsilon = 0.1

Q = np.zeros((state_size, action_size))

if random.uniform(0,1) < epsilon:
else:
    

