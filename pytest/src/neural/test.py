#!/usr/bin/env python
import numpy as np;

sizes = [2, 3, 1]
biases = [np.random.randn(y, 1) for y in sizes[1:]]
print("biases")
print(biases)
zipped = zip(sizes[:-1], sizes[1:])
print("zipped")
for z in zipped:
    print(z)
weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
print("weights")
print(weights)
for x, y in zip(sizes[:-1], sizes[1:]):
    print(x, y)
