import numpy as np
import theano
import theano.tensor as T

from theano.tensor.nnet import sigmoid
from theano.tensor import shared_randomstreams


def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1 - p_dropout, size=layer.shape)
    return layer * T.cast(mask, theano.config.floatX)


n_in = 784
n_out = 100
mini_batch_size = 10
p_dropout = 0

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
output = sigmoid((1 - p_dropout) * T.dot(inpt, w) + b)
theano.printing.pydotprint(output, "./a.png")
y_out = T.argmax(output, axis=1)
inpt_dropout = dropout_layer(x.reshape((mini_batch_size, n_in)), p_dropout)

print(theano.printing.debugprint(y_out))
theano.printing.pydotprint(y_out, "./b.png")
# output = sigmoid([.1])
# f = theano.function([], output)  
# f()
# print(theano.printing.debugprint(output))
# theano.printing.pydotprint(output, "./a.png")
# print(f())
