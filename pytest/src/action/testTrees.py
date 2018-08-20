import trees
import treePlotter

myDat, labels = trees.createDataSet()
print(labels)
print(myDat)
myTree = treePlotter.retrieveTree(0)
print(myTree)
print(trees.classify(myTree, labels, [1, 0]))
