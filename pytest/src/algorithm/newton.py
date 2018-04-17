#!/usr/bin/env python
# 本文件演示下用牛顿法求方程的解

# 求解方程 fx = x * x - 9
def F(x):
    return x * x - 9;


# 导数
def F_prime(x):
    return 2 * x


# 得到下一个X
def Xn(x):
    return x - F(x) / F_prime(x)


# 差值
def E(xn, x):
    return xn - x


# 给x一个预设值
x = 10;

xn = Xn(x)
e = 0.000001

i = 0
while abs(E(xn, x)) > e:
    i = i + 1
    x = xn
    xn = Xn(x)
    print("Epoch {0}:x {1} xn {2}".format(i, x, xn))

