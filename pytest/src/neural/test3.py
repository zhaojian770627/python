#!/usr/bin/env python
import random;

import numpy as np;

aw = np.array([[1, 2], [4, 5], [7, 8]])
# A1[...]
# B1[...]
ax = np.array([[1, 2, 3], [1, 2, 3]])
wx = np.dot(aw, ax)
print(wx)
b = np.array([1, 1, 1]).transpose();
z = wx + b
print(z)
