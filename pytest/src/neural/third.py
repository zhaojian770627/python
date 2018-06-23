import numpy as np
import pylab


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


x = np.array(range(1, 100)) / 100

b = -40 
w = 500

# 一个神经元的输出
# y = sigmoid(w * x + b)
# 
# pylab.plot(x, y, 'r-', label=u'b -40 w 100')
# pylab.legend()

# 两个神经元
s1 = 0
s2 = .2
h1 = -.3
h2 = .3

s3 = .2
s4 = .4
h3 = -1.3
h4 = 1.3

s5 = .4
s6 = .6
h5 = -.5
h6 = .5

s7 = .6
s8 = .8
h7 = 1.3
h8 = -1.3

s9 = .8
s10 = 1.0
h9 = -.2
h10 = .2

b1 = -s1 * w;
b2 = -s2 * w;
b3 = -s3 * w;
b4 = -s4 * w;
b5 = -s5 * w;
b6 = -s6 * w;
b7 = -s7 * w;
b8 = -s8 * w;
b9 = -s9 * w;
b10 = -s10 * w;

y2 = sigmoid(w * x + b1) * h1 + sigmoid(w * x + b2) * h2 \
+ sigmoid(w * x + b3) * h3 + sigmoid(w * x + b4) * h4 \
+ sigmoid(w * x + b5) * h5 + sigmoid(w * x + b6) * h6 \
+ sigmoid(w * x + b7) * h7 + sigmoid(w * x + b8) * h8 \
+ sigmoid(w * x + b9) * h9 + sigmoid(w * x + b10) * h10

pylab.plot(x, y2, 'b-', label=u'4 ')
pylab.legend()

pylab.show()
