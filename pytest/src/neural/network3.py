"""network3.py
~~~~~~~~~~~~~~

A Theano-based program for training and running simple neural
networks.

Supports several layer types (fully connected, convolutional, max
pooling, softmax), and activation functions (sigmoid, tanh, and
rectified linear units, with more easily added).

When run on a CPU, this program is much faster than network.py and
network2.py.  However, unlike network.py and network2.py it can also
be run on a GPU, which makes it faster still.

Because the code is based on Theano, the code is different in many
ways from network.py and network2.py.  However, where possible I have
tried to maintain consistency with the earlier programs.  In
particular, the API is similar to network2.py.  Note that I have
focused on making the code simple, easily readable, and easily
modifiable.  It is not optimized, and omits many desirable features.

This program incorporates ideas from the Theano documentation on
convolutional neural nets (notably,
http://deeplearning.net/tutorial/lenet.html ), from Misha Denil's
implementation of dropout (https://github.com/mdenil/dropout ), and
from Chris Olah (http://colah.github.io ).

Written for Theano 0.6 and 0.7, needs some changes for more recent
versions of Theano.

"""

#### Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
# from theano.tensor.nnet import conv
from theano.tensor.nnet import conv2d
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal.pool import pool_2d


# Activation functions for neurons
def linear(z): return z


def ReLU(z): return T.maximum(0.0, z)


from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

#### Constants
GPU = True
if GPU:
    print("Trying to run under a GPU.  If this is not desired, then modify " + \
        "network3.py\nto set the GPU flag to False.")
    try: theano.config.device = 'gpu'
    except: pass  # it's already set
    theano.config.floatX = 'float32'
else:
    print("Running with a CPU.  If this is not desired, then the modify " + \
        "network3.py to set\nthe GPU flag to True.")


#### Load the MNIST data
def load_data_shared(filename="/home/zj/git/neural-networks-and-deep-learning/data/mnist.pkl.gz"):
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='bytes')
    f.close()

    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.

        """
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")

    return [shared(training_data), shared(validation_data), shared(test_data)]


#### Main class used to construct and train networks
class Network(object):

    def __init__(self, layers, mini_batch_size):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.

        """
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.matrix("x")
        self.y = T.ivector("y")
        init_layer = self.layers[0]
        # 设置第一层的输入及输出
        # 将输⼊ self.x 传了两次，可能会以两种⽅式（有dropout 和⽆ dropout）使⽤⽹络
        init_layer.set_inpt(self.x, self.x, self.mini_batch_size)
        for j in range(1, len(self.layers)):
            prev_layer, layer = self.layers[j - 1], self.layers[j]
            # 设置第一层之后的输入为上一层的输出
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout, self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, lmbda=0.0):
        """Train the network using mini-batch stochastic gradient descent."""
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        # compute number of minibatches for training, validation and testing
        num_training_batches = size(training_data) / mini_batch_size
        num_validation_batches = size(validation_data) / mini_batch_size
        num_test_batches = size(test_data) / mini_batch_size

        # define the (regularized) cost function, symbolic gradients, and updates
        # L2 规范化
        l2_norm_squared = sum([(layer.w ** 2).sum() for layer in self.layers])
        
        # cost定义为最后一层即总 SoftmaxLayer 的代价函数
        cost = self.layers[-1].cost(self) + \
               0.5 * lmbda * l2_norm_squared / num_training_batches
        
        # 计算梯度及导数，params包含各层的w b,注意是不同的w b
        # grads[0] 对第一层w b的导数 grads[1] 对第二层w b的导数,虽然名字一样，但不是同样的东西
        # theano会生成整体的表达式 以此类推
        grads = T.grad(cost, self.params)
        
        updates = [(param, param - eta * grad)
                   for param, grad in zip(self.params, grads)]

        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.
        i = T.lscalar()  # mini-batch index
        # 定义函数
        train_mb = theano.function(
            [i], cost, updates=updates,
            givens={
                self.x:
                training_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size],
                self.y:
                training_y[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            })
        validate_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                validation_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size],
                self.y:
                validation_y[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            })
        test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                test_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size],
                self.y:
                test_y[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            })
        self.test_mb_predictions = theano.function(
            [i], self.layers[-1].y_out,
            givens={
                self.x:
                test_x[i * self.mini_batch_size: (i + 1) * self.mini_batch_size]
            })
        # Do the actual training
        best_validation_accuracy = 0.0
        for epoch in range(epochs):
            for minibatch_index in range(int(num_training_batches)):
                iteration = num_training_batches * epoch + minibatch_index
                if iteration % 1000 == 0:
                    print("Training mini-batch number {0}".format(iteration))
                cost_ij = train_mb(minibatch_index)
                if (iteration + 1) % num_training_batches == 0:
                    validation_accuracy = np.mean(
                        [validate_mb_accuracy(j) for j in range(int(num_validation_batches))])
                    print("Epoch {0}: validation accuracy {1:.2%}".format(
                        epoch, validation_accuracy))
                    if validation_accuracy >= best_validation_accuracy:
                        print("This is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        if test_data:
                            test_accuracy = np.mean(
                                [test_mb_accuracy(j) for j in range(int(num_test_batches))])
                            print('The corresponding test accuracy is {0:.2%}'.format(
                                test_accuracy))
        print("Finished training network.")
        print("Best validation accuracy of {0:.2%} obtained at iteration {1}".format(
            best_validation_accuracy, best_iteration))
        print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))

#### Define layer types


# 卷积及混合层，这里组合到了一起
class ConvPoolLayer(object):
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.

    """

    def __init__(self, filter_shape, image_shape, poolsize=(2, 2),
                 activation_fn=sigmoid):
        """`filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and the
        filter width.

        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.

        `poolsize` is a tuple of length 2, whose entries are the y and
        x pooling sizes.

        """
        # filter_shape=(20, 1, 5, 5)
        self.filter_shape = filter_shape
        # image_shape=(mini_batch_size, 1, 28, 28)
        self.image_shape = image_shape
        # poolsize=(2, 2)
        self.poolsize = poolsize
        # activation_fn=sigmoid --> 1/(1+exp(-z))
        self.activation_fn = activation_fn
        # initialize weights and biases
        # n_out=125
        # 20 * (5 * 5 ) / 4
        n_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize))
        # w 为 20*1*5*5 矩阵 
        # 20 个特征 每个特征只对应一个输入映射 过滤器为5*5 局部感受野
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(1.0 / n_out), size=filter_shape),
                dtype=theano.config.floatX),
            borrow=True)
        self.b = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                dtype=theano.config.floatX),
            borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        # 10 * 1 *28 * 28 每批10个样本，每个样本一副图像，每个图像为28*28像素的数据
        self.inpt = inpt.reshape(self.image_shape)
        # 可以将 w 看做是待训练的滤波器
        conv_out = conv2d(
            input=self.inpt, filters=self.w, filter_shape=self.filter_shape,
            input_shape=self.image_shape)
        pooled_out = pool_2d(
            input=conv_out, ws=self.poolsize, ignore_border=True)
        self.output = self.activation_fn(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output_dropout = self.output  # no dropout in the convolutional layers


# 全连接的网络
class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        # 弃权技术用，用于过度拟合化
        self.p_dropout = p_dropout
        # Initialize weights and biases
        # w 为 n_in 行 n_out 列的正态随机分布矩阵
        # n_in=784, 输入784 n_out=100 100 个神经元
        # 最终得到 784 * 100 的矩阵
        self.w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0 / n_out), size=(n_in, n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)
        # n_out=100 个 正态分布变量
        # b 为 100*1 的矩阵
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype=theano.config.floatX),
            name='b', borrow=True)
        # w b放在 params 中
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        # 设定输入变量，每批量10个样本
        # 形成 10 * 784 列的矩阵 这里 mini_batch_size 10 n_in 784 
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        # z=@(w,b,x) w*x+b
        # 设定 output= sigmoid( (1-p_dropout) *  dot(inpt,w) +b )
        # 1-p_dropout 不是很明白
        # [ 10 * 784 ] DOT [ 784 * 100 ] --> [10*100]  
        self.output = self.activation_fn(
            (1 - self.p_dropout) * T.dot(self.inpt, self.w) + self.b)
        # 返回axis轴上最大值的索引
        self.y_out = T.argmax(self.output, axis=1)
        # inpt_dropout 为 10 * 784 矩阵 对输入进行弃权
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        # 对于弃权后的输入计算 z=@(w,b,x) w*x+b 
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))


# softmax层，这一层始终放到最后
class SoftmaxLayer(object):

    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        # 设定输入变量，每批量10个样本
        # 形成 10 * 100 列的矩阵 这里 mini_batch_size 10 n_in 100 
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax((1 - self.p_dropout) * T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        # softmax 分类函数
        self.output_dropout = softmax(T.dot(self.inpt_dropout, self.w) + self.b)

    # 平均代价
    def cost(self, net):
        "Return the log-likelihood cost."
        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))


#### Miscellanea
def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]


# 生成弃权矩阵表达式
def dropout_layer(layer, p_dropout):
    # 产生随机数
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    # 生成非0即1的矩阵mask，矩阵的规模和layer相同
    mask = srng.binomial(n=1, p=1 - p_dropout, size=layer.shape)
    # 掩码矩阵和输入矩阵相乘，得到弃权后的矩阵
    return layer * T.cast(mask, theano.config.floatX)
