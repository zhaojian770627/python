import matplotlib
from numpy import *

import kNN
import matplotlib.pyplot as plt

# group, labels = kNN.createDataSet()
# print(kNN.classify0([0, 0], group, labels, 3))
datingDataMat, datingLabels = kNN.file2matrix('/home/zj/datingTestSet2.txt')
fig = plt.figure()
ax = fig.add_subplot(111)
# 参数 x,y,伸缩率,颜色
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
plt.title('Sample')  # 显示图表标题
plt.xlabel('x')  # x轴名称
plt.ylabel('y')  # y轴名称
plt.show()
