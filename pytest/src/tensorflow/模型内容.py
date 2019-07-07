import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
savedir = "/home/zj/temp/tensorflow/save/"
print_tensors_in_checkpoint_file(savedir + "linermodel.cpkt",None,True)