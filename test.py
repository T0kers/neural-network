import numpy as np


a = np.array([1, 2, 3, 4])

print(np.eye(4) * 0 - a)

print(a[:, np.newaxis] * (np.eye(4) - a))