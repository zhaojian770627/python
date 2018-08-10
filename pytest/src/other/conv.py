import numpy as np
x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
h = np.array([[1, 1], [1, 1]])
import scipy.signal
print(scipy.signal.convolve(x, h))  # 卷积运算
