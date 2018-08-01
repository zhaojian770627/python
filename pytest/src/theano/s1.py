import theano.tensor as T
from theano import function
from theano import pp
 
x = T.dmatrices('x')
y = T.dmatrices('y')
z = x + y
f = function([x, y], z)

print(f([[1,2],[3,4]],[[10,20],[30,40]]))