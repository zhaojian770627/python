import pickle
import gzip

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
# from theano.tensor.nnet import conv
from theano.tensor.nnet import conv2d
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal.pool import pool_2d

d = T.dmatrix("d")
x = np.random.random((28, 28))
print(x)
