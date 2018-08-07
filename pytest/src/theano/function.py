import numpy as np
import theano
import theano.tensor as T

from theano import function

state = theano.shared(np.array(0, dtype=np.float64), 'state')  # inital state = 0
inc = T.scalar('inc', dtype=state.dtype)
accumulator = theano.function([inc], state, updates=[(state, state + inc)])

print(state.get_value())
accumulator(10)
print(state.get_value())

tmp_func = state * 2 + inc 
a = T.scalar(dtype=state.dtype)
# a -> state
skip_shared = function([inc, a], tmp_func, givens={state: a})  # temporarily use a's value for the state
print(skip_shared(2, 3))