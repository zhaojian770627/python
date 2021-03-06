import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data


def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx - w):idx]) / w for idx, val in enumerate(a)]


tf.random.set_random_seed(1)

mnist = input_data.read_data_sets("/home/zj/temp/MNIST_data/", one_hot=True)

X = tf.placeholder(tf.float32, [784, None])
Y = tf.placeholder(tf.float32, [10, None])

W = tf.Variable(tf.random_normal([10, 784]))
b = tf.Variable(tf.zeros([10, 1]))

pred = tf.nn.softmax(tf.matmul(W, X) + b, axis=0)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(pred), reduction_indices=0))

learning_rate = .01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

training_epochs = 25
batch_size = 100
display_step = 1
saver = tf.train.Saver()
model_path = "/home/zj/temp/tensorflow/save/521model.ckpt"

plotdata = { "batchsize":[], "loss":[] }

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            _, c = sess.run([optimizer, cost], feed_dict={X:batch_xs.T, Y:batch_ys.T})
            avg_cost += c / total_batch

        if(epoch + 1) % display_step == 0:
            plotdata["batchsize"].append(epoch)
            plotdata["loss"].append(avg_cost)
            print ("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
            
    print(" Finished!")

    plotdata["avgloss"] = moving_average(plotdata["loss"], w=25)
    plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
    plt.xlabel("Minibatch number")
    plt.ylabel("Loss")
    plt.title("Minibatch run vs. Training loss")
    
    plt.show()

    # 测试model
    print(pred.shape)
    correct_prediction = tf.equal(tf.argmax(pred, 0), tf.argmax(Y, 0))
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print ("Accuracy:", accuracy.eval({X: mnist.test.images.T, Y: mnist.test.labels.T}))
    
    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)

