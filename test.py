import numpy as np


# a = np.array([1, 2, 3, 4])

# print(np.eye(4) * 0 - a)

# print(a[:, np.newaxis] * (np.eye(4) - a))


a = np.array([
    [1, 2, 3],
    [3, 4, 5]
])

b = np.array([10, 11])

d = {"a": a, "b": b}

print(a)

d["a"][0, 0] = 100

print(d)
print(a)