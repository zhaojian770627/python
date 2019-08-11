import numpy as np
# from scipy.misc import imsave
import pylab

filename = '/home/zj/temp/cifar-10-batches-py/data_batch_1'

# filename = '/home/zj/tensorflow_datasets/cifar10/1.0.2/cifar10-train.tfrecord-00000-of-00010'

bytestream = open(filename, "rb")
buf = bytestream.read(10000 * (1 + 32 * 32 * 3))
bytestream.close()

data = np.frombuffer(buf, dtype=np.uint8)

data = data.reshape(10000, 1 + 32 * 32 * 3)
labels_images = np.hsplit(data, [1])
labels = labels_images[0].reshape(10000)
images = labels_images[1].reshape(10000, 32, 32, 3)

img = np.reshape(images[0], (3, 32, 32))
img = img.transpose(1, 2, 0)

print(labels[0])
pylab.imshow(img)
pylab.show()
