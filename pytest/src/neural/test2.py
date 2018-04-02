#!/usr/bin/env python
import numpy as np;


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def feedforward(biases, weights, a):
    for b, w in zip(biases, weights):
        a = sigmoid(np.dot(w, a) + b)
    return a

    
data = [1, 2, 3]
ary = np.array(data)
result = sigmoid(ary)
print(result)
