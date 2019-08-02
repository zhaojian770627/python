import tensorflow as tf

# [batch, in_height, in_width, in_channels]
# [训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]  

input = tf.Variable(tf.constant(1.0, shape=[1, 5, 5, 1])) 
input2 = tf.Variable(tf.constant(1.0, shape=[1, 5, 5, 2]))
input3 = tf.Variable(tf.constant(1.0, shape=[1, 4, 4, 1])) 

# [filter_height, filter_width, in_channels, out_channels] 
# [卷积核的高度，卷积核的宽度，图像通道数，卷积核个数] 
filter1 = tf.Variable(tf.constant([-1.0, 0, 0, -1], shape=[2, 2, 1, 1]))

# 1个通道输入，生成1个feature map
op1 = tf.nn.conv2d(input, filter1, strides=[1, 2, 2, 1], padding='SAME')  

init = tf.global_variables_initializer()  
with tf.Session() as sess:  
    sess.run(init)  

    print("op1:\n", sess.run([op1, filter1]))  # 1-1  后面补0
    print("------------------")
