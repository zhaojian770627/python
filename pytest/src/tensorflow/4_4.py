import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx - w):idx]) / w for idx, val in enumerate(a)]


np.random.seed(1)
tf.random.set_random_seed(1)
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * .3
# plt.plot(train_X, train_Y, 'ro', label='Original data')
# plt.legend()
# plt.show()

X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

w = tf.Variable(tf.random_normal([1]), name='w')
b = tf.Variable(tf.zeros([1]), name='b')
z = tf.add(tf.multiply(w, X), b) 
cost = tf.reduce_mean(tf.square(Y - z))
leaning_rate = .01
optimizer = tf.train.GradientDescentOptimizer(leaning_rate).minimize(cost)
init = tf.global_variables_initializer()
training_epochs = 20
display_step = 2

saver=tf.train.Saver() # 生成saver
savedir="/home/zj/temp/tensorflow/save/"

with tf.Session() as sess:
    sess.run(init)
    plotdata = { "batchsize":[], "loss":[] }
    writer = tf.summary.FileWriter('/home/zj/graph', sess.graph)
    
    for epoch in range(training_epochs):
        for x, y in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X:x, Y:y})
            writer.close()
        if epoch % display_step == 0:
            loss = sess.run(cost, feed_dict={X:train_X, Y:train_Y})
            print("Epoch:", epoch + 1, "cost=", loss, "W=", sess.run(w), "b=", sess.run(b))
            if not (loss == "NA"):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)
    
    print(" Finished!")
    saver.save(sess, savedir+"linermodel.cpkt")
    print("cost=", sess.run(cost, feed_dict={X:train_X, Y:train_Y}), "w=", sess.run(w), "b=", sess.run(b))

    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(w) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
    
    plotdata["avgloss"] = moving_average(plotdata["loss"])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
    plt.xlabel("Minibatch number")
    plt.ylabel("Loss")
    plt.title("Minibatch run vs. Training loss")
    
    plt.show()
    print("x=.2,z=", sess.run(z, feed_dict={X:0.2}))
    
# 加载模型
with tf.Session() as sess2:
    sess2.run(tf.global_variables_initializer())     
    saver.restore(sess2, savedir +"linermodel.cpkt")
    print ("x=0.2，z=", sess2.run(z, feed_dict={X: 0.2}))
