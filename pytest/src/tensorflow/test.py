import numpy as np
import tensorflow as tf

c = np.array([[-6.8694515],
 [-7.8843102],
 [ 2.6226492]]
    )

input_features = tf.placeholder(tf.float32, [3, 1])

output = tf.nn.softmax(input_features, axis=0)
a1 = tf.argmax(output, axis=0)
print(output)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    o, a = sess.run([output, a1], feed_dict={input_features:c})
    
    print(o)
    print(a)
