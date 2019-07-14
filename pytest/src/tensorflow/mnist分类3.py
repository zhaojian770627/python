import tensorflow as tf
import pylab

from tensorflow.examples.tutorials.mnist import input_data

tf.random.set_random_seed(1)

mnist = input_data.read_data_sets("/home/zj/temp/MNIST_data/", one_hot=True)

X = tf.placeholder(tf.float32, [784, None])  # mnist data维度 28*28=784
Y = tf.placeholder(tf.float32, [10, None])  # 0-9 数字=> 10 classes

# Set model weights
W = tf.Variable(tf.random_normal([10, 784]))
b = tf.Variable(tf.zeros([10, 1]))

pred = tf.nn.softmax(tf.matmul(W, X) + b, axis=0)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(pred), reduction_indices=0))

learning_rate = .01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

training_epochs = 25
batch_size = 100
display_step = 1
saver = tf.train.Saver
model_path = "/home/zj/temp/log/521model.ckpt"

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs_t = batch_xs.T
            batch_ys_t = batch_ys.T
            _, c = sess.run([optimizer, cost], feed_dict={X:batch_xs.T, Y:batch_ys.T})
            avg_cost += c / total_batch

        if(epoch + 1) % display_step == 0:
            print ("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
            
    print(" Finished!")
    
    
