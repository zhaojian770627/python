import kNN

group, labels = kNN.createDataSet()
print(kNN.classify0([0, 0], group, labels, 3))

datingDataMat, datingLabels = kNN.file2matrix('/home/zj/datingTestSet.txt')
print(datingLabels)