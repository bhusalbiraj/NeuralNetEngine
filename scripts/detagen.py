from nnfs.datasets import spiral_data
import numpy as np

X, y = spiral_data(samples=100, classes=3)

np.savetxt("X.csv", X, delimiter=",")
np.savetxt("y.csv", y, fmt="%d", delimiter=",")