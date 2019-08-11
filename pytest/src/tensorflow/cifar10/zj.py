import numpy as np
import pylab


def mypick(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


filename = '/home/zj/temp/cifar-10-batches-py/data_batch_1'
data = mypick(filename)
images = data[b'data']

img = np.reshape(images[0], (3, 32, 32))
img = img.transpose(1, 2, 0)

pylab.imshow(img)
pylab.show()

