import tensorflow as tf
import numpy as np

np.random.seed(1)
tf.random.set_random_seed(1)

learning_rate = 1e-4  # 输入层节点个数
n_input = 2
n_label = 1
n_hidden = 2  # 隐藏层节点个数

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

loss = tf.reduce_mean((y_pred - y) ** 2)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# 生成数据
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [[0], [1], [1], [0]]
X = np.array(X).astype('float32')
Y = np.array(Y).astype('int16')

# 加载
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 训练
    for i in range(10000):
        sess.run([train_step, y_pred, layer_2], feed_dict={x:X.T, y:Y.T})
    # 计算预测值
    print(sess.run(y_pred, feed_dict={x:X.T}))
    # 输出：已训练100000次
       
    # 查看隐藏层的输出
    print(sess.run(layer_1, feed_dict={x:X.T}))
