import numpy as np

a = np.array([[0], [0]])
b = np.array([[1,0,1,1,0]])

print(a @ b.T)
print(np.dot(a, b.T))
print(a * b)
print(np.maximum(0, a))