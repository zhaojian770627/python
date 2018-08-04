import numpy as np
import theano
import theano.tensor as T

x = T.dmatrix('x')  
y = T.dmatrix('y')  
z = x + y  
print(theano.printing.debugprint(z))
theano.printing.pydotprint(z, "./a.png")