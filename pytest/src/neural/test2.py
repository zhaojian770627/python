#!/usr/bin/env python
import numpy as np;


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def feedforward(biases, weights, a):
    for b, w in zip(biases, weights):
        a = sigmoid(np.dot(w, a) + b)
    return a

sizes = [2, 3, 1]
biases = [np.random.randn(y, 1) for y in sizes[1:]]
weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

print("biases")
print(biases)
print("zipped")
zipped = zip(sizes[:-1], sizes[1:])
for z in zipped:
    print(z)
print("weights")
print(weights)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
for b, w in zip(biases, weights):
    print("--------------b-------------")
    print(b)
    print("--------------b.shape-------------")
    print(b.shape)
    c=np.zeros(b.shape)
    print(c)
    print("--------------w-------------")
    print(w)
    print("--------------w.shape-------------")
    print(w.shape)
    print("---------------------------")