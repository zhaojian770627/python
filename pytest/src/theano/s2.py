import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import sigmoid

n_in = 784
n_out = 100
w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0 / n_out), size=(n_in, n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)
b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype=theano.config.floatX),
            name='b', borrow=True)
x = T.matrix("x")
output = sigmoid([.5])
f = theano.function([], output)  
f()
print(theano.printing.pp(output))
print(f())