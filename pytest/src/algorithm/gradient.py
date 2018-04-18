#!/usr/bin/env python
# 本文件演示下用梯度求函数的极小值

# 求解方程 fx = x * x + 9


def F(x):
    return x * x + x;


# 导数
def F_prime(x):
    return 2 * x + 1


# 得到下一个X
def Xn(x):
    return x - eta * F_prime(x)


def DELTA(x):
    return -F_prime(x) * eta


# 差值
def E(xn, x):
    return abs(xn - x)


# 给x一个预设值
# 学习速率
eta = 0.001
x = 10
xn = 10
e = 0.00000000001

i = 0
while True:
    deltaX = DELTA(x)
    if(abs(deltaX) < e):
        break
    i = i + 1
#     if(i > 200):
#         break ;
    xn = x + deltaX
    x = xn
    print("Epoch {0}:x {1} xn {2}".format(i, x, xn))

x = xn
print("最优解 {0}".format(x))
