import numpy as np
import theano
import theano.tensor as T

from theano import function

n_in = 784
n_out = 100
mini_batch_size = 2
A = np.random.rand(mini_batch_size, n_in)
B = A.reshape(mini_batch_size, n_in)
C = np.asarray(np.random.normal(loc=0.0, scale=np.sqrt(1.0 / n_out), size=(n_in, n_out)))
w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0 / n_out), size=(n_in, n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)
print(type(w))
i = T.lscalar()
y = i * 2
f = function([i], y)
print(f([1]))