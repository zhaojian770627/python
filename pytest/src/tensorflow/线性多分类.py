import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from matplotlib.colors import colorConverter, ListedColormap 
    
# 对于上面的fit可以这么扩展变成动态的
from sklearn.preprocessing import OneHotEncoder

np.random.seed(10)
tf.random.set_random_seed(10)


def onehot(y, start, end):
    ohe = OneHotEncoder()
    a = np.linspace(start, end - 1, end - start)
    b = np.reshape(a, [-1, 1]).astype(np.int32)
    ohe.fit(b)
    c = ohe.transform(y).toarray()  
    return c 


def generate(sample_size, num_classes, diff, regression=False):
    np.random.seed(10)
    mean = np.random.randn(2)
    cov = np.eye(2)  
    
    # len(diff)
    samples_per_class = int(sample_size / num_classes)

    X0 = np.random.multivariate_normal(mean, cov, samples_per_class)
    Y0 = np.zeros(samples_per_class)
    
    for ci, d in enumerate(diff):
        X1 = np.random.multivariate_normal(mean + d, cov, samples_per_class)
        Y1 = (ci + 1) * np.ones(samples_per_class)
    
        X0 = np.concatenate((X0, X1))
        Y0 = np.concatenate((Y0, Y1))
        # print(X0, Y0)
  
    if regression == False:  # one-hot  0 into the vector "1 0
        Y0 = np.reshape(Y0, [-1, 1])        
        # print(Y0.astype(np.int32))
        Y0 = onehot(Y0.astype(np.int32), 0, num_classes)
        # print(Y0)
    X, Y = shuffle(X0, Y0)
    # print(X, Y)
    return X, Y 


def showimage(X, Y):
    aa = [np.argmax(l) for l in Y]
    colors = ['r' if l == 0 else 'b' if l == 1 else 'y' for l in aa[:]]
    plt.scatter(X[:, 0], X[:, 1], c=colors)
    plt.xlabel("Scaled age (in yrs)")
    plt.ylabel("Tumor size (in cm)")
    plt.show()

    
input_dim = 2
num_classes = 3 
X, Y = generate(2000, num_classes, [[3.0], [3.0, 0]], False)
# showimage(X, Y)

lab_dim = num_classes
# tf Graph Input
input_features = tf.placeholder(tf.float32, [input_dim, None ])
input_labels = tf.placeholder(tf.float32, [lab_dim, None])

# Set model weights
W = tf.Variable(tf.random_normal([lab_dim, input_dim ]), name="weight")
b = tf.Variable(tf.zeros([lab_dim, 1]), name="bias")

output = tf.matmul(W, input_features) + b

z = tf.nn.softmax(output)

a1 = tf.argmax(tf.nn.softmax(output), axis=0)  # 按列找出最大索引，生成数组
b1 = tf.argmax(input_labels, axis=0)

err = tf.count_nonzero(a1 - b1)  # 两个数组相减，不为0的就是错误个数

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=input_labels, logits=output)
loss = tf.reduce_mean(cross_entropy)  # 对交叉熵取均值很有必要

optimizer = tf.train.AdamOptimizer(0.04)  # 尽量用这个--收敛快，会动态调节梯度
train = optimizer.minimize(loss)  # let the optimizer train

maxEpochs = 50
minibatchSize = 25

# 启动session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(maxEpochs):
        sumerr = 0
        for i in range(np.int32(len(Y) / minibatchSize)):
            x1 = X[i * minibatchSize:(i + 1) * minibatchSize, :]
            y1 = Y[i * minibatchSize:(i + 1) * minibatchSize, :]

            _, lossval, outputval, errval = sess.run([train, loss, output, err], feed_dict={input_features: x1.T, input_labels:y1.T})
            sumerr = sumerr + (errval / minibatchSize)
    
        print ("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(lossval), "err=", sumerr / (np.int32(len(Y) / minibatchSize)))
