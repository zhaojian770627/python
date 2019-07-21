import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from matplotlib.colors import colorConverter, ListedColormap 

# 对于上面的fit可以这么扩展变成动态的
from sklearn.preprocessing import OneHotEncoder


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
  
    if regression == False:  # one-hot  0 into the vector "1 0
        Y0 = np.reshape(Y0, [-1, 1])        
        # print(Y0.astype(np.int32))
        Y0 = onehot(Y0.astype(np.int32), 0, num_classes)
        # print(Y0)
    X, Y = shuffle(X0, Y0)
    # print(X, Y)
    return X, Y   


# Ensure we always get the same amount of randomness
np.random.seed(10)
tf.random.set_random_seed(1)

input_dim = 2
num_classes = 4 
X, Y = generate(320, num_classes, [[3.0, 0], [3.0, 3.0], [0, 3.0]], True)

# 将其中的两类数据合并
Y = Y % 2

xr = []
xb = []
for(l, k) in zip(Y[:], X[:]):
    if l == 0.0 :
        xr.append([k[0], k[1]])        
    else:
        xb.append([k[0], k[1]])
xr = np.array(xr)
xb = np.array(xb)      
plt.scatter(xr[:, 0], xr[:, 1], c='r', marker='+')
plt.scatter(xb[:, 0], xb[:, 1], c='b', marker='o')

plt.show() 
# 转换为列
Y = np.reshape(Y, [-1, 1])

learning_rate = 1e-4
n_input = 2
n_label = 1
# n_hidden = 2  # 欠拟合
n_hidden = 200

# 输入
x = tf.placeholder(tf.float32, [n_input, None ]) 
# 输出
y = tf.placeholder(tf.float32, [n_label, None])

# 权重
weights = {
# 第一层 2*2
    'h1': tf.Variable(tf.truncated_normal([n_hidden, n_input], stddev=0.1)),
# 第二层 1*2
    'h2': tf.Variable(tf.random_normal([n_label, n_hidden ], stddev=0.1))
    } 
# 偏置
biases = {
# 第一层 2*1
    'h1': tf.Variable(tf.zeros([n_hidden, 1])),
# 第二层 1*1
    'h2': tf.Variable(tf.zeros([n_label, 1]))
    }  

layer_1 = tf.nn.relu(tf.add(tf.matmul(weights['h1'], x), biases['h1']))
layer_2 = tf.add(tf.matmul(weights['h2'], layer_1), biases['h2'])

# Leaky relus 在ReLU基础上， 保留一部分负值， 让x为负时乘0.01， 即Leaky relus对负信号不
# 是一味地拒绝， 而是缩小。
y_pred = tf.maximum(layer_2, 0.01 * layer_2)
# y_pred = tf.sigmoid(layer_2)

loss = tf.reduce_mean((y_pred - y) ** 2)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# 加载
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 训练
    for i in range(20000):
        _, loss_val = sess.run([train_step, loss], feed_dict={x:X.T, y:Y.T})
        
        if i % 1000 == 0:
            print ("Step:", i, "Current loss:", loss_val)

    xr = []
    xb = []
    for(l, k) in zip(Y[:], X[:]):
        if l == 0.0 :
            xr.append([k[0], k[1]])        
        else:
            xb.append([k[0], k[1]])
    xr = np.array(xr)
    xb = np.array(xb)      
    plt.scatter(xr[:, 0], xr[:, 1], c='r', marker='+')
    plt.scatter(xb[:, 0], xb[:, 1], c='b', marker='o')

    nb_of_xs = 200
    xs1 = np.linspace(-3, 10, num=nb_of_xs)
    xs2 = np.linspace(-3, 10, num=nb_of_xs)
    xx, yy = np.meshgrid(xs1, xs2)  # create the grid
    # Initialize and fill the classification plane
    classification_plane = np.zeros((nb_of_xs, nb_of_xs))
    for i in range(nb_of_xs):
        for j in range(nb_of_xs):
            # classification_plane[i,j] = nn_predict(xx[i,j], yy[i,j])
            classification_plane[i, j] = sess.run(y_pred, feed_dict={x: [
                                                                           [ xx[i, j]],
                                                                           [ yy[i, j]]
                                                                        ]})
            classification_plane[i, j] = int(classification_plane[i, j])

    # Create a color map to show the classification colors of each grid point
    cmap = ListedColormap([
            colorConverter.to_rgba('r', alpha=0.30),
            colorConverter.to_rgba('b', alpha=0.30)])
    # Plot the classification plane with decision boundary and input samples
    plt.contourf(xx, yy, classification_plane, cmap=cmap)
    plt.show() 
    
    xTrain, yTrain = generate(12, num_classes, [[3.0, 0], [3.0, 3.0], [0, 3.0]], True)
    yTrain = yTrain % 2
    # colors = ['r' if l == 0.0 else 'b' for l in yTrain[:]]
    # plt.scatter(xTrain[:,0], xTrain[:,1], c=colors)
    
    xr = []
    xb = []
    for(l, k) in zip(yTrain[:], xTrain[:]):
        if l == 0.0 :
            xr.append([k[0], k[1]])        
        else:
            xb.append([k[0], k[1]])
    xr = np.array(xr)
    xb = np.array(xb)      
    plt.scatter(xr[:, 0], xr[:, 1], c='r', marker='+')
    plt.scatter(xb[:, 0], xb[:, 1], c='b', marker='o')
    
    yTrain = np.reshape(yTrain, [-1, 1])           
    print ("loss:\n", sess.run(loss, feed_dict={x: xTrain.T, y: yTrain.T}))          
    
    nb_of_xs = 200
    xs1 = np.linspace(-1, 8, num=nb_of_xs)
    xs2 = np.linspace(-1, 8, num=nb_of_xs)
    xx, yy = np.meshgrid(xs1, xs2)  # create the grid
    # Initialize and fill the classification plane
    classification_plane = np.zeros((nb_of_xs, nb_of_xs))
    for i in range(nb_of_xs):
        for j in range(nb_of_xs):
            # classification_plane[i,j] = nn_predict(xx[i,j], yy[i,j])
            classification_plane[i, j] = sess.run(y_pred, feed_dict={x: [
                                                                            [ xx[i, j]],
                                                                            [ yy[i, j]]
                                                                        ]
                                                                    })
            classification_plane[i, j] = int(classification_plane[i, j])
    
    # Create a color map to show the classification colors of each grid point
    cmap = ListedColormap([
            colorConverter.to_rgba('r', alpha=0.30),
            colorConverter.to_rgba('b', alpha=0.30)])
    # Plot the classification plane with decision boundary and input samples
    plt.contourf(xx, yy, classification_plane, cmap=cmap)
    plt.show()   
