import tensorflow as tf

# 导入 MINST 数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/zj/temp/MNIST_data/", one_hot=True)

tf.random.set_random_seed(1)

# 参数设置
learning_rate = 0.001
training_epochs = 25
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256  # 1st layer number of features
n_hidden_2 = 256  # 2nd layer number of features
n_input = 784  # MNIST data 输入 (img shape: 28*28)
n_classes = 10  # MNIST 列别 (0-9 ，一共10类)

# tf Graph input
x = tf.placeholder("float", [n_input, None])
y = tf.placeholder("float", [n_classes, None])


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(weights['h1'], x), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(weights['h2'], layer_1), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(weights['out'], layer_2) + biases['out']
    return out_layer


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([ n_hidden_1, n_input])),
    'h2': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_classes, n_hidden_2 ]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1, 1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2, 1])),
    'out': tf.Variable(tf.random_normal([n_classes, 1]))
}

pred = multilayer_perceptron(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y, dim=0))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # 启动循环开始训练
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        # 遍历全部数据集
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x.T,
                                                          y: batch_y.T})
            # Compute average loss
            avg_cost += c / total_batch
        # 显示训练中的详细信息
        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch + 1), "cost=", \
                "{:.9f}".format(avg_cost))
    print (" Finished!")

    # 测试 model
    correct_prediction = tf.equal(tf.argmax(pred, 0), tf.argmax(y, 0))
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    a = sess.run([accuracy], feed_dict={x: mnist.test.images.T, y: mnist.test.labels.T})
    print("Accuracy:", a)
