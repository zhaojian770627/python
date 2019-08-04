import tensorflow as tf
import math

# [batch, in_height, in_width, in_channels]
# [训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]  

input = tf.Variable(tf.constant(1.0, shape=[1, 5, 5, 1])) 
input2 = tf.Variable(tf.constant(1.0, shape=[1, 5, 5, 2]))
input3 = tf.Variable(tf.constant(1.0, shape=[1, 4, 4, 1])) 


def calcpadding(type, in_height, in_width, filter_height, filter_width, strides_height, strides_width):
    output_height = 0;
    output_width = 0;
    if(type == "VALID"):
        print("VALID-----------------------------------------")
        output_height = math.ceil((in_height - filter_width + 1) / strides_height)
        output_width = math.ceil((in_width - filter_width + 1) / strides_width)
        print("output_height:" + str(output_height))
        print("output_width:" + str(output_width))
    else:
        print("SAME-----------------------------------------")
        output_height = math.ceil(in_height / strides_height)
        output_width = math.ceil(in_width / strides_width)
        print("output_height:" + str(output_height))
        print("output_width:" + str(output_width))
        
    pad_height = max((output_height - 1) * strides_height + filter_height - in_height, 0)
    pad_width = max((output_width - 1) * strides_width + filter_width - in_width, 0)
    pad_top = math.floor(pad_height / 2)
    pad_bottom = pad_height - pad_top
    pad_left = math.floor(pad_width / 2)
    pad_right = pad_width - pad_left
    
    print("pad_height:" + str(pad_height))
    print("pad_width:" + str(pad_width))
    print("pad_top:" + str(pad_top))
    print("pad_bottom:" + str(pad_bottom))
    print("pad_left:" + str(pad_left))
    print("pad_right:" + str(pad_right))
    

# [filter_height, filter_width, in_channels, out_channels] 
# [卷积核的高度，卷积核的宽度，图像通道数，卷积核个数] 
filter1 = tf.Variable(tf.constant([-1.0, 0, 0, -1], shape=[2, 2, 1, 1]))
filter2 = tf.Variable(tf.constant([-1.0, 0, 0, -1, -1.0, 0, 0, -1], shape=[2, 2, 1, 2])) 
# 1个通道输入，生成1个feature map
op1 = tf.nn.conv2d(input, filter1, strides=[1, 2, 2, 1], padding='SAME')  
op2 = tf.nn.conv2d(input, filter2, strides=[1, 2, 2, 1], padding='SAME')  # 1个通道输入，生成2个feature map

calcpadding("SAME", 5, 5, 2, 2, 2, 2)

init = tf.global_variables_initializer()  
with tf.Session() as sess:  
    sess.run(init)  

    print("op1:\n", sess.run([op1, filter1]))  # 1-1  后面补0
    print("------------------")
    print("op2:\n", sess.run([op2, filter2]))  # 1-2多卷积核 按列取
