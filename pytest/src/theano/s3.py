from theano import function
from theano import pp
import theano
from theano.tensor import shared_randomstreams
from theano.tensor.nnet import sigmoid
import numpy as np
import theano.tensor as T

n_in = 784
n_out = 100
mini_batch_size = 10
p_dropout = 0.2
x = T.matrix("x")
# 10*784 矩阵
layer = x.reshape((mini_batch_size, n_in))
# layer = x.reshape((10, 10))
srng = shared_randomstreams.RandomStreams(np.random.RandomState(0).randint(999999))
mask = srng.binomial(n=1, p=1 - p_dropout, size=layer.shape)
out = layer * T.cast(mask, theano.config.floatX)
print(theano.printing.debugprint(mask))
A = np.random.RandomState(0).rand(10, 784)
print("----------------A---------------")
print(A)
f = function([x], out)
print("----------------F---------------")
print(f(A))
