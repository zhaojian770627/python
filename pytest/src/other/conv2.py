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
filter_shape = (1, 1, 5, 5)
image_shape = (1, 1, 6, 6)
poolsize = (2, 2)
x = T.matrix("x")
n_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize))
a = np.asarray(np.random.randint(low=1, high=2, size=(5, 5)))
print(a)
w = theano.shared(a.reshape(filter_shape), borrow=True)
inpt = x.reshape(image_shape)
conv_out = conv2d(input=inpt, filters=w, filter_shape=filter_shape, input_shape=image_shape)
print(theano.printing.debugprint(conv_out))
theano.printing.pydotprint(conv_out, "/home/zj/a.png")
f = theano.function([x], conv_out)
rin = np.random.randint(low=1, high=5, size=(6, 6))
print(rin)
print(f(rin))
