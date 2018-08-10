import gzip
import pickle

import theano
from theano.tensor import shared_randomstreams
from theano.tensor.nnet import conv2d
from theano.tensor.nnet import softmax
from theano.tensor.signal.pool import pool_2d

import numpy as np
import theano.tensor as T

# 卷积及混合的测试
# Third-party libraries
# from theano.tensor.nnet import conv
d = T.dmatrix("d")
filter_shape = (1, 1, 2, 2)
image_shape = (1, 1, 3, 3)
poolsize = (2, 2)
x = T.matrix("x")
print("Filter-------------")
ft = np.asarray(np.random.randint(low=1, high=2, size=(filter_shape[2:])))
print(ft)
shareft = theano.shared(ft.reshape(filter_shape), borrow=True)
inpt = x.reshape(image_shape)
conv_out = conv2d(input=inpt, filters=shareft, filter_shape=filter_shape, input_shape=image_shape)
f = theano.function([x], conv_out)
rin = np.random.randint(low=1, high=5, size=(image_shape[2:]))
print("Input------------------------")
print(rin)
print(f(rin))
