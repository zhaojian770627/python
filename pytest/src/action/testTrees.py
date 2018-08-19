import trees
import treePlotter

myDat, labels = trees.createDataSet()
print(labels)
print(myDat)
# print(trees.splitDataSet(myDat, 0, 1))
# print(trees.chooseBestFeatureToSplit(myDat))
# print(trees.createTree(myDat, labels))
# treePlotter.createPlot()
treePlotter.retrieveTree(1)