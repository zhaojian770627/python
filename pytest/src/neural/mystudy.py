#!/usr/bin/env python
import pickle
import gzip
from PIL import Image

import numpy as np

import network


def load_data():
    f = gzip.open('/home/zj/git/neural-networks-and-deep-learning/data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='bytes')
    f.close()
    return (training_data, validation_data, test_data)


def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def showimage(image):
    image.resize((28, 28))
    im = Image.fromarray((image * 256).astype('uint8'))
    im.show()


# training_data, validation_data, test_data = load_data_wrapper()
# print(training_data[0])
# print(training_data[0])
# net = network.Network([784, 30, 10])
# net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
# net.printbw()

ti = [
        [[0], [0]],
        [[0], [1]],
        [[1], [0]],
        [[1], [1]],
        [[0], [0]],
        [[1], [1]],
        
        [[0], [0]],
        [[0], [1]],
        [[1], [0]],
        [[1], [1]],
        [[0], [0]],
        [[1], [1]],
        
        [[0], [0]],
        [[0], [1]],
        [[1], [0]],
        [[1], [1]],
        [[0], [0]],
        [[1], [1]],
        
        [[0], [0]],
        [[0], [1]],
        [[1], [0]],
        [[1], [1]],
        [[0], [0]],
        [[1], [1]],
        
        [[0], [0]],
        [[0], [1]],
        [[1], [0]],
        [[1], [1]],
        [[0], [0]],
        [[1], [1]],
        
        [[0], [0]],
        [[0], [1]],
        [[1], [0]],
        [[1], [1]],
        [[0], [0]],
        [[1], [1]],
        
        [[0], [0]],
        [[0], [1]],
        [[1], [0]],
        [[1], [1]],
        [[0], [0]],
        [[1], [1]],
        
        [[0], [0]],
        [[0], [1]],
        [[1], [0]],
        [[1], [1]],
        [[0], [0]],
        [[1], [1]],
        
        [[0], [0]],
        [[0], [1]],
        [[1], [0]],
        [[1], [1]],
        [[0], [0]],
        [[1], [1]],
        
        [[0], [0]],
        [[0], [1]],
        [[1], [0]],
        [[1], [1]],
        [[0], [0]],
        [[1], [1]]
     ]
tr = [
        [[0]],
        [[1]],
        [[1]],
        [[0]],
        [[0]],
        [[0]],
        
        [[0]],
        [[1]],
        [[1]],
        [[0]],
        [[0]],
        [[0]],
        
        [[0]],
        [[1]],
        [[1]],
        [[0]],
        [[0]],
        [[0]],
        
        [[0]],
        [[1]],
        [[1]],
        [[0]],
        [[0]],
        [[0]],
        
        [[0]],
        [[1]],
        [[1]],
        [[0]],
        [[0]],
        [[0]],
        
        [[0]],
        [[1]],
        [[1]],
        [[0]],
        [[0]],
        [[0]],
        
        [[0]],
        [[1]],
        [[1]],
        [[0]],
        [[0]],
        [[0]],
        
        [[0]],
        [[1]],
        [[1]],
        [[0]],
        [[0]],
        [[0]],
        
        [[0]],
        [[1]],
        [[1]],
        [[0]],
        [[0]],
        [[0]],
        
        [[0]],
        [[1]],
        [[1]],
        [[0]],
        [[0]],
        [[0]]
    ]

tti = [
        [[0], [0]],
        [[0], [1]],
        [[1], [0]],
        [[1], [1]],
        [[0], [0]],
        [[1], [1]]
     ]
ttr = [
        [[0]],
        [[1]],
        [[1]],
        [[0]],
        [[0]],
        [[0]]
    ]

tdi = np.array(ti);
tdr = np.array(tr);

td = list(zip(tdi, tdr))

ttd = list(zip(np.array(tti), np.array(ttr)))

net = network.Network([2, 100, 1]);
net.SGD(td, 30, 2, 3.0,ttd)
