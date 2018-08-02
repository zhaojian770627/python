#!/usr/bin/env python
import theano;
import theano.tensor as T

x = T.fscalar('x')
y = T.fscalar('y')
z = x ** 2 + y ** 2
grads = theano.grad(z, [x, y])
f=theano.function([x,y],grads[1])
print(f(1,2))