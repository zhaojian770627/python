#!/usr/bin/env python
import random;
import numpy as np;
import network2
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network2.Network([784, 30, 10])
test_cost, test_accuracy, training_cost, training_accuracy \
 = net.SGD(training_data, 30, 10, .05, lmbda=50.0, evaluation_data=None, monitor_evaluation_accuracy=False, monitor_training_cost=True)
print(training_cost)
