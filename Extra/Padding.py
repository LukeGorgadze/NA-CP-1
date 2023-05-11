import numpy as np

a = np.array([[1,2,3],[1,2,3],[1,2,3]])
b = np.array([[2,3,4],[2,3,4],[2,3,6]])

padding = 2
padded_a = np.zeros((a.shape[0] + 2 * padding, a.shape[1] + 2*padding))
padded_a[padding:-padding, padding:-padding] = a
print(padded_a[padding:-4])