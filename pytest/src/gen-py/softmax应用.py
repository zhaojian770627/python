import tensorflow as tf

labels = [[0, 0, 1], [0, 1, 0]]
logits = [[2, .5, 6], [.1, 0, 3]]

logits_scaled = tf.nn.softmax(logits)
logits_scaled2 = tf.nn.softmax(logits_scaled)

result1 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
result2 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits_scaled)
result3 = -tf.reduce_sum(labels * tf.log(logits_scaled), 1)

with tf.Session() as sess:
    print("scaled=", sess.run(logits_scaled))
    print("scaled2=", sess.run(logits_scaled2))
    
    print("rel1=", sess.run(result1), "\n")
    print("rel2=", sess.run(result2), "\n")
    print("rel3=", sess.run(result3))

labels = [[0.4, 0.1, 0.5], [0.3, 0.6, 0.1]]
result4 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
with tf.Session() as sess:
    print ("rel4=", sess.run(result4), "\n") 

labels = [2, 1]  # 其实是0 1 2 三个类。等价 第一行 001 第二行 010
result5 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
with tf.Session() as sess:
    print ("rel5=", sess.run(result5), "\n")

loss = tf.reduce_mean(result1)
with tf.Session() as sess:
    print ("loss=", sess.run(loss))
    
labels = [[0, 0, 1], [0, 1, 0]]    
loss2 = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(logits_scaled), 1))
with tf.Session() as sess:
    print ("loss2=", sess.run(loss2))
