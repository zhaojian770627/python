import numpy as np
import theano
import theano.tensor as T

from theano.tensor.nnet import sigmoid

n_in = 784
n_out = 100
mini_batch_size = 10

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
inpt = x.reshape((mini_batch_size, n_in))
print(theano.printing.debugprint(inpt))
theano.printing.pydotprint(inpt, "./a.png")
# output = sigmoid([.1])
# f = theano.function([], output)  
# f()
# print(theano.printing.debugprint(output))
# theano.printing.pydotprint(output, "./a.png")
# print(f())
