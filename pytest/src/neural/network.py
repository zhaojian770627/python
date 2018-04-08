#!/usr/bin/env python
import random
import numpy as np;

# 神经元的输入函数
# 1/( 1 + exp(− ∑wx  − b))
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# 上面函数导数，求导暂时还不会
def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))

# 神经网络的代码
class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes);
        self.sizes = sizes
        # 从第二层开始
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # x 截止到最后一个(包含)
        # y 从第一个开始算
        # 表示每个后续的神经元有几个前面的神经元输入与之连接，
        # 就是表示每个神经元与之对应的输入的权重
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

    # training_data 是⼀个 (x, y) 元组的列表，表⽰训练输⼊和其对应的期望输出。
    # epochs 迭代期数量
    # mini_batch_size 采样时的⼩批量数据的⼤⼩
    # eta 学习速率 η
    # test_data 测试数据 程序会在每个训练器后评估⽹络
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}:{1}/{2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))
            
    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        self.weights = [w - (eta / len(mini_batch)) * nw
                      for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                     for b, nb in zip(self.biases, nabla_b)]

    # 计算∇C
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # 前馈
        activation = x
        # list to store all the activations, layer by layer
        activations = [x]
        # list to store all the z vectors, layer by layer
        zs = []
        for b, w in zip(self.biases, self.weights):
            # 计算每层的z变量即 wx+b (x 为上一层的输出结果)
            z = np.dot(w, activation) + b
            zs.append(z)
            # 计算输出 按 1/( 1 + exp(− ∑wx  − b))
            activation = sigmoid(z)
            activations.append(activation)
        # 计算增量 activations 最后加入的就是最终的神经元的输出
        # -1 表示最后一个神经元的输出
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        # 转置
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        for l in range(2, self.num_layers):
            z = zs[-l]
            # 导数
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        
        return (nabla_b, nabla_w)
    
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    # 计算计算值和真实值的差值，用来
    def cost_derivative(self, output_activations, y):
        return (output_activations - y)
