#!/usr/bin/env python
import random
import numpy as np;
from numpy.core.tests.test_mem_overlap import xrange


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes);
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
       
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            # 对每一层神经网络，默认从第2层开始
            # w 为每一层权重矩阵，每一行对应一个神经元的输入，每一行的每一列表示上层的每个输入
            # 如以下表示 这一层有三个神经元，每个神经元有两个输入的权重，输入肯定是[x1,x2]的转置
            # 相乘后是一个R3向量，然后和b相加，b是R3向量，最后得到一个R3向量
            # 这就是 ∑w + b 的结果
            # [ w11 w12]  |x1|  |b1|   |a1|
            # [ w21 w22]× |  |+ |b2| = |a2|
            # [ w31 w32]  |x2|  |b3|   |a3|
            # 
            # 上述结果经sigmoid运算,得到这一层神经元输出
            a = sigmoid(np.dot(w, a) + b)
            # 将结果用于下一层w正好对应来源层的输出，如果下一层
            #                |a1|
            # [ w11,w12,w13]*|a2|+|b| 
            #                |a3|
        
        # 最终返回总输出
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}:{1}/{2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))
            
print("abc")
