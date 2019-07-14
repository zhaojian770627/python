import tensorflow as tf
import pylab 

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

# 读取模型
print("Starting 2nd session...")
with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())
    # Restore model weights from previously saved model
    saver.restore(sess, model_path)
    
    # 测试 model
    correct_prediction = tf.equal(tf.argmax(pred, 0), tf.argmax(Y, 0))
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print ("Accuracy:", accuracy.eval({X: mnist.test.images.T, Y: mnist.test.labels.T}))
    
    output = tf.argmax(pred, 1)
    batch_xs, batch_ys = mnist.train.next_batch(2)
    outputval, predv = sess.run([output, pred], feed_dict={X: batch_xs.T})
    print(outputval, predv, batch_ys.T)

    im = batch_xs[0]
    im = im.reshape(-1, 28)
    pylab.imshow(im)
    pylab.show()
    
    im = batch_xs[1]
    im = im.reshape(-1, 28)
    pylab.imshow(im)
    pylab.show()
