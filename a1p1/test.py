import numpy as np

example = np.array([
    [1, 0],
    [0, 1],
    [1, 1]
])
print(np.where(example > 0))