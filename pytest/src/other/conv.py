import numpy as np
x = np.array([1, 2, 3])
h = np.array([4, 5, 6])
import scipy.signal
print(scipy.signal.convolve(x, h))  # 卷积运算
