import numpy as np

rewards = [np.array([1,2]),np.array([3,4])]

rewards = np.concatenate(rewards)

print(rewards)